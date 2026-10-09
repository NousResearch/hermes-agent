"""Archived connection-lifecycle slice from pre-Phase-3 hermes_cli.sqlite_safe_read.

Used only to reproduce a still-running old module during a source update.
Its registry, tracker, and lock intentionally remain in the legacy module.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_live_lock = threading.RLock()
_live_connections: dict[str, int] = {}


def _key(path: Path | str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def _track_key(key: str, delta: int = 1) -> None:
    remaining = _live_connections.get(key, 0) + delta
    if remaining > 0:
        _live_connections[key] = remaining
    else:
        _live_connections.pop(key, None)


def untrack_connection(path: Path | str) -> None:
    with _live_lock:
        _track_key(_key(path), -1)


def has_live_connection(path: Path | str) -> bool:
    key = _key(path)
    with _live_lock:
        return (key in _live_connections
                or any(key.endswith(suffix) and key[:-len(suffix)] in _live_connections
                       for suffix in ("-wal", "-shm")))


def read_header_bytes_preopen(path: Path | str, *, length: int = 100) -> bytes | None:
    with _live_lock:
        if has_live_connection(path):
            return None
        with open(path, "rb") as handle:
            return handle.read(length)


class _TrackingMixin:
    _hermes_tracked_path: str | None = None

    def close(self) -> None:
        with _live_lock:
            path = getattr(self, "_hermes_tracked_path", None)
            super().close()
            if path is not None:
                self._hermes_tracked_path = None
                untrack_connection(path)


class TrackedConnection(_TrackingMixin, sqlite3.Connection):
    pass


def connect_tracked(path: Path | str, **kwargs) -> sqlite3.Connection:
    with _live_lock:
        conn = sqlite3.connect(str(path), factory=TrackedConnection, **kwargs)
        row = conn.execute("PRAGMA database_list").fetchone()
        if row and row[2]:
            key = _key(row[2])
            conn._hermes_tracked_path = key
            _track_key(key)
        return conn
