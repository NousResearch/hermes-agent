"""Utility module for sample_repo fixture."""
import json


def helper():
    """A helper function."""
    return "helper_output"


def load_json(filepath: str) -> dict:
    """Load JSON from a file."""
    with open(filepath, "r") as f:
        return json.load(f)
