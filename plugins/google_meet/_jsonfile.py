"""Tiny JSON file helpers shared by the bot, process manager, node registry and node server."""

from __future__ import annotations

import contextlib
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


def write_json_atomic(path: Path, data: Any, mode: Optional[int] = None) -> None:
    """Write ``json.dumps(data, indent=2)`` via a ``.json.tmp`` sibling + rename; *mode* (e.g.
    ``0o600``) is applied to the temp file so the final file never exists with looser perms."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if mode is not None:
        with contextlib.suppress(OSError, NotImplementedError):  # best-effort on non-POSIX filesystems
            tmp.chmod(mode)
    tmp.replace(path)
