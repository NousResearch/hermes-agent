"""Cross-device open-tab list for one Hermes profile home.

Desktop used to remember the tab strip only in the client's localStorage, keyed
by connection. A second device talking to the same backend therefore opened an
empty strip even though the sessions themselves were still in ``state.db``.

This file is the shared list: ordered stored-session ids plus dock placement.
Runtime ids stay off disk — they are process-scoped and must be re-resumed.
Last-writer-wins is revision-checked so a stale client cannot clobber a newer
device. An empty list is a real value (the user closed every tab), not a
missing file.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

FILENAME = "desktop-open-tabs.json"
MAX_TILES = 40
_MAX_ID = 200
_ALLOWED_DIRS = frozenset({"bottom", "center", "left", "right", "top"})

_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}


class RevisionConflict(Exception):
    """``base_revision`` did not match the file. ``current`` is the winner."""

    def __init__(self, current: dict):
        super().__init__("open-tabs revision conflict")
        self.current = current


def empty_document() -> dict:
    return {"revision": 0, "updated_at": None, "tiles": []}


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


def _path(home: Path) -> Path:
    return Path(home) / FILENAME


def _clip(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > _MAX_ID:
        return None
    return text


def normalize_tiles(raw: Any) -> list[dict]:
    """Keep a bounded, durable placement list. Unknown fields are dropped."""
    if not isinstance(raw, list):
        return []
    tiles: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        stored = _clip(item.get("storedSessionId"))
        if not stored or stored in seen:
            continue
        seen.add(stored)
        tile: dict[str, Any] = {"storedSessionId": stored}
        direction = item.get("dir")
        if isinstance(direction, str) and direction in _ALLOWED_DIRS:
            tile["dir"] = direction
        anchor = _clip(item.get("anchor"))
        if anchor:
            tile["anchor"] = anchor
        before = item.get("before")
        if before is None and "before" in item:
            tile["before"] = None
        else:
            clipped = _clip(before)
            if clipped:
                tile["before"] = clipped
        tiles.append(tile)
        if len(tiles) >= MAX_TILES:
            break
    return tiles


def load(home: Path) -> dict:
    path = _path(home)
    with _lock_for(path):
        return _read_unlocked(path)


def save(home: Path, tiles: Any, base_revision: int) -> dict:
    """Write ``tiles`` if ``base_revision`` is still current. Else conflict."""
    if isinstance(base_revision, bool) or not isinstance(base_revision, int) or base_revision < 0:
        raise ValueError("base_revision must be a non-negative int")
    path = _path(home)
    with _lock_for(path):
        current = _read_unlocked(path)
        if current["revision"] != base_revision:
            raise RevisionConflict(current)
        document = {
            "revision": current["revision"] + 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "tiles": normalize_tiles(tiles),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
        return document


def _read_unlocked(path: Path) -> dict:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return empty_document()
    if not isinstance(parsed, dict):
        return empty_document()
    revision = parsed.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        revision = 0
    updated = parsed.get("updated_at")
    if not isinstance(updated, str) or not updated:
        updated = None
    return {
        "revision": revision,
        "updated_at": updated,
        "tiles": normalize_tiles(parsed.get("tiles")),
    }
