import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_absolute_error , mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r"https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
print(data.head())
print(data.sample())
print(data.describe())
data.isnull().sum()

corr=data.corr()
#sns.heatmap(corr)
import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame = data, x="Sales",
                    y="TV", size="TV", trendline="ols")
figure.show()

print(corr["Sales"].sort_values(ascending=False))

x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

model1=LinearRegression()
model1.fit(xtrain,ytrain)
features = np.array([[230.1, 37.8, 69.2]])
print(model1.predict(features))
print(model1.score(xtest, ytest))

model2=DecisionTreeRegressor(max_depth=10)
model2.fit(xtrain,ytrain)
print(model2.score(xtest, ytest))