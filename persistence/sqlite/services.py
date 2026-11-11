"""Store services that answer without a healthy store: the corruption latch and the readiness probe.

Neither creates, opens writable, migrates or repairs a database; the probe reads the schema through a ``mode=ro``
connection only when the file already exists.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from hermes_state_health import STORAGE_CORRUPT, note_storage_error, storage_state

from persistence import StorageProbe, StorageState

from ._backend import register_repository

__all__ = ["SQLiteStorageHealth", "probe_state_db"]


def probe_state_db(path: Path) -> StorageProbe:
    """Read-only schema probe plus the process-wide corruption latch (``persistence.sqlite._state.health``).

    The schema read only catches an unreadable header or schema; damage deeper in the file surfaces when a reader
    or writer touches it, and those publish into the latch."""
    if not path.exists():
        return StorageProbe(status="ok", detail="not initialized")
    if storage_state(path) == STORAGE_CORRUPT:
        return StorageProbe(status="degraded", detail=STORAGE_CORRUPT)
    try:
        # ``closing`` is required: sqlite3's context manager only commits or rolls back, never closes.
        with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=1.0)) as conn:
            # A readiness probe must never compete with normal state writers.
            conn.execute("PRAGMA query_only = ON")
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
        return StorageProbe(status="ok", detail=None)
    except Exception as exc:  # health: allow BLE001 -- any probe failure is reported, never raised, as the readiness probe does
        if note_storage_error(path, exc):
            return StorageProbe(status="degraded", detail=STORAGE_CORRUPT)
        return StorageProbe(status="degraded", detail=type(exc).__name__)


class SQLiteStorageHealth:
    """``StorageHealth`` over the process-wide latch, bound to one ``state.db``."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def storage_state(self) -> StorageState:
        corrupt = storage_state(self._path) == STORAGE_CORRUPT
        return StorageState.CORRUPT if corrupt else StorageState.OK

    def probe(self) -> StorageProbe:
        return probe_state_db(self._path)


register_repository("health", lambda store: SQLiteStorageHealth(store.store_path))
