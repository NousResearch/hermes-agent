"""Tiny JSON read helper shared by the bot, process manager, node registry and node server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def read_json(path: Path) -> Optional[Any]:
    """Parsed JSON from *path*, or None when missing/unreadable/malformed."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
