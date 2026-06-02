"""Pure utility module - not executable."""

import os
import json


def load_config(path):
    with open(path) as f:
        return json.load(f)


def main_util():
    """Utility named with main prefix - not an entrypoint."""
    return load_config("config.json")
