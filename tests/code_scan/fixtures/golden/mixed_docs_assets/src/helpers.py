"""Helper utilities for loading assets."""
import os
import json


def load_assets(root: str) -> list:
    """Load asset catalog from the project root."""
    catalog_path = os.path.join(root, "assets", "data", "catalog.json")
    if os.path.isfile(catalog_path):
        with open(catalog_path) as f:
            return json.load(f).get("items", [])
    return []
