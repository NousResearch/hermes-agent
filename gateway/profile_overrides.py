"""Persistent per-source profile overrides for the gateway.

Lets a chat (and optionally a thread) pin itself to a named profile at
runtime via a slash command (``/profile set <name>``), without restarting
the gateway.

The override map is keyed by the same scope identifiers the multiplexing
router uses (``platform``, ``chat_id``, ``thread_id``), so it composes
cleanly with :mod:`gateway.profile_routing` — an explicit override always
wins over a ``profile_routes`` rule. The lookup is hierarchical: a
thread override beats a chat override beats no override.

State lives in ``<hermes_root>/profile_overrides.json`` (atomic writes).
This file is shared across profiles by design — it is the runtime switch
table, not profile-local state.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from hermes_constants import get_default_hermes_root
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_OVERRIDES_FILE = "profile_overrides.json"
_WRITE_LOCK = threading.RLock()


def _overrides_path() -> Path:
    return get_default_hermes_root() / _OVERRIDES_FILE


def _read_map() -> dict:
    path = _overrides_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("Failed to read profile overrides, starting empty", exc_info=True)
        return {}


def _write_map(data: dict) -> None:
    path = _overrides_path()
    atomic_json_write(path, data, sort_keys=True)


@contextmanager
def _locked_update() -> Iterator[None]:
    """Serialize profile-override read-modify-write updates.

    The in-process lock protects concurrent picker callbacks. The sibling lock
    file extends that protection across gateway processes, while
    :func:`atomic_json_write` uses a unique temporary file for each writer so a
    concurrent replace can never collide on a shared ``.json.tmp`` pathname.
    """
    path = _overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")

    with _WRITE_LOCK:
        with lock_path.open("a+b") as lock_fh:
            if os.name == "nt":
                import msvcrt

                # msvcrt.locking locks from the current file position and
                # requires a byte to exist at that position.
                lock_fh.seek(0, os.SEEK_END)
                if lock_fh.tell() == 0:
                    lock_fh.write(b"\0")
                    lock_fh.flush()
                lock_fh.seek(0)
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    import msvcrt

                    lock_fh.seek(0)
                    msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _key_for(platform: str, chat_id: str, thread_id: Optional[str]) -> str:
    """Build the override key for a (platform, chat, thread) scope."""
    pid = str(platform)
    cid = str(chat_id)
    tid = str(thread_id) if thread_id else ""
    if tid:
        return f"{pid}:{cid}:{tid}"
    return f"{pid}:{cid}"


def set_override(platform: str, chat_id: str, profile: str, thread_id: Optional[str] = None) -> None:
    """Pin ``platform:chat_id[:thread_id]`` to ``profile``.

    ``profile`` is stored normalized (lowercase; ``default`` special-cased)
    by the caller; we don't import profiles here to keep this module
    dependency-light.
    """
    with _locked_update():
        data = _read_map()
        data[_key_for(platform, chat_id, thread_id)] = profile
        _write_map(data)


def clear_override(platform: str, chat_id: str, thread_id: Optional[str] = None) -> bool:
    """Remove an override. Returns True if something was removed."""
    with _locked_update():
        data = _read_map()
        key = _key_for(platform, chat_id, thread_id)
        if key in data:
            del data[key]
            _write_map(data)
            return True
        return False


def resolve_override(
    platform: str, chat_id: str, thread_id: Optional[str] = None
) -> Optional[str]:
    """Return the pinned profile for this scope, or None.

    Thread-specific override wins over chat-level override.
    """
    data = _read_map()
    if thread_id:
        hit = data.get(_key_for(platform, chat_id, thread_id))
        if hit:
            return hit
    return data.get(_key_for(platform, chat_id, None))


def list_overrides() -> dict:
    """Return the full override map (read-only copy)."""
    return dict(_read_map())
