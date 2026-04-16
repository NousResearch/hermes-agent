import sys
import os

# Add the langchain/libs/core directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'langchain', 'libs', 'core'))

from pydantic import BaseModel
from typing import TypeVar, Generic
from langchain_core.tools.base import get_all_basemodel_annotations

# Test with a simple model
class SimpleModel(BaseModel):
    name: str
    age: int

# Test with a generic model
T = TypeVar('T')

class GenericModel(BaseModel, Generic[T]):
    value: T

# Test with inheritance
class ChildModel(SimpleModel):
    email: str

# Test with generic inheritance
class ChildGenericModel(GenericModel[str]):
    extra: str

# Test the function
print("Testing get_all_basemodel_annotations...")

# Test 1: Simple model
print("\n1. SimpleModel:")
annotations = get_all_basemodel_annotations(SimpleModel)
print(annotations)

# Test 2: Generic model
print("\n2. GenericModel[str]:")
annotations = get_all_basemodel_annotations(GenericModel[str])
print(annotations)

# Test 3: Inherited model
print("\n3. ChildModel:")
annotations = get_all_basemodel_annotations(ChildModel)
print(annotations)

# Test 4: Generic inherited model
print("\n4. ChildGenericModel:")
annotations = get_all_basemodel_annotations(ChildGenericModel)
print(annotations)

print("\nAll tests completed successfully!")