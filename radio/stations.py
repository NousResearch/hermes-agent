"""Curated station list loader.

Loads from built-in radio/stations.yaml, overridden by user's
~/.hermes/radio/stations.yaml if it exists.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

BUILTIN_PATH = Path(__file__).parent / "stations.yaml"
USER_PATH = Path(os.path.expanduser("~/.hermes/radio/stations.yaml"))


def load_stations() -> List[Dict[str, Any]]:
    """Load curated stations. User file overrides built-in."""
    path = USER_PATH if USER_PATH.exists() else BUILTIN_PATH
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return data.get("stations", [])
    except Exception:
        return []
