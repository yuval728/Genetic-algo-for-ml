# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
import os

# %%
telescope=pd.read_csv('Data/telescope_data.csv')
telescope.drop(telescope.columns[0],axis=1,inplace=True)
# telescope.head()

# %%
telescope_shuffle=telescope.iloc[np.random.permutation(len(telescope))]
telescope=telescope_shuffle.reset_index(drop=True)


# %%
telescope['class']=telescope['class'].map({'g':0,'h':1})

# %%
tele_class = telescope['class'].values
tele_features = telescope.drop('class',axis=1).values


# %%
training_data, testing_data, training_classes, testing_classes = train_test_split(tele_features, tele_class, test_size=0.25, random_state=42, stratify=tele_class)

# %%
tpot = TPOTClassifier(generations=5,verbosity=2)
tpot.fit(training_data, training_classes)

# %%
print(tpot.score(testing_data, testing_classes))

# %%

os.makedirs('Output',exist_ok=True)
tpot.export('Output/tpot_pipeline.py')

# %%



