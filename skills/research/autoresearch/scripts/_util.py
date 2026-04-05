#!/usr/bin/env python3
"""Shared utilities for autoresearch helper scripts."""
import json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def hermes_home():
    """Return the active Hermes home directory, respecting HERMES_HOME."""
    return os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))


def atomic_write(path, data):
    """Write JSON data atomically via tempfile + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, str(path))
    except Exception:
        os.unlink(tmp)
        raise


def read_json(path):
    """Read a JSON file, returning {} on missing/corrupt files."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
