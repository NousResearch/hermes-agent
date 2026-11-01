"""Small JSON cache primitives shared by model catalog siblings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _read_json_cache(path: Path, *, errors=Exception) -> Optional[dict]:
    """Load a JSON-object cache file; None when missing, unreadable, or not a dict."""
    try:
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except errors:
        return None
    return data if isinstance(data, dict) else None


def _write_json_cache(path: Path, data: Any, **dump_kwargs: Any) -> None:
    """Atomically persist a cache file (creating parents). Raises on failure — callers decide
    whether a failed cache write is worth logging."""
    from utils import atomic_json_write
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(path.parent)
    atomic_json_write(path, data, **dump_kwargs)
