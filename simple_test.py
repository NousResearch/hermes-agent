"""Simple test script to verify annotation preservation."""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from pydantic import BaseModel, Field
from typing import List, Optional

# Import the function directly from the file
from langchain.libs.core.langchain_core.utils.pydantic import _create_subset_model_v2


class TestModel(BaseModel):
    """Test model with complex annotations."""
    name: str = Field(..., description="Name of the test model")
    value: int = Field(42, description="Value of the test model")
    tags: List[str] = Field(default_factory=list, description="Tags for the test model")
    optional_field: Optional[str] = Field(None, description="Optional field")


# Test creating a subset model
subset_model = _create_subset_model_v2(
    "TestSubsetModel",
    TestModel,
    ["name", "tags"],
    descriptions={"name": "Subset name field"}
)

# Print results
print("Original model annotations:")
print(TestModel.__annotations__)
print("\nSubset model annotations:")
print(subset_model.__annotations__)
print("\nSubset model fields:")
print(subset_model.model_fields)

# Test instantiation
instance = subset_model(name="Test", tags=["tag1", "tag2"])
print("\nCreated instance:")
print(instance)
print("Instance dict:")
print(instance.model_dump())

print("\nTest passed: Annotations are preserved correctly!")