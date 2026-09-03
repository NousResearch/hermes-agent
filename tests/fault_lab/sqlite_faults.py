"""Real cross-process SQLite write-lock contention.

Spawns a REAL separate process that opens the target database file,
takes an exclusive write lock, and holds it — so a write attempt in the
test process hits a genuine ``sqlite3.OperationalError: database is
locked`` from actual OS-level file locking, not a mocked connection.
"""

from __future__ import annotations

import multiprocessing
import sqlite3
import time
from pathlib import Path
from typing import Optional

__all__ = ["LockHolder"]


def _hold_exclusive_lock(db_path: str, ready: "multiprocessing.synchronize.Event",
                          release: "multiprocessing.synchronize.Event") -> None:
    conn = sqlite3.connect(db_path, timeout=1)
    conn.execute("BEGIN EXCLUSIVE")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fault_lab_lock_holder (id INTEGER)"
    )
    ready.set()
    release.wait(timeout=30)
    conn.rollback()
    conn.close()


class LockHolder:
    """Context manager: a real subprocess holds an exclusive lock on ``db_path``."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = str(db_path)
        self._ctx = multiprocessing.get_context("spawn")
        self._ready = self._ctx.Event()
        self._release = self._ctx.Event()
        self._proc: Optional[multiprocessing.process.BaseProcess] = None

    def __enter__(self) -> "LockHolder":
        self._proc = self._ctx.Process(
            target=_hold_exclusive_lock,
            args=(self._db_path, self._ready, self._release),
            daemon=True,
        )
        self._proc.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("fault_lab: lock-holder subprocess never acquired the lock")
        # Small buffer so the exclusive transaction is fully committed
        # from SQLite's perspective before the caller attempts a write.
        time.sleep(0.05)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._release.set()
        if self._proc is not None:
            self._proc.join(timeout=10)
