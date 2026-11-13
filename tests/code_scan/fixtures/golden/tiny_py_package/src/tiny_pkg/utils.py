"""Utility module for tiny package."""
import os
import json


def format_msg(name: str = "world") -> str:
    """Format a greeting message."""
    return f"Hello, {name}!"


def load_config(path: str) -> dict:
    """Load a JSON configuration file."""
    with open(path, "r") as f:
        return json.load(f)
