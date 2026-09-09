"""SQLite connection and transaction helpers shared by cron ledgers."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


def open_ledger(path: Path) -> sqlite3.Connection:
    """Open a profile-local ledger DB, creating its cron directory securely."""
    from cron.jobs import _ensure_cron_dir

    _ensure_cron_dir(path.parent)
    return sqlite3.connect(path, timeout=5)


def prepare_ledger(
    conn: sqlite3.Connection, *, db_label: str, synchronous_full: bool = True
) -> None:
    """Configure row access, busy timeout, WAL, and optional full synchronization."""
    from hermes_state_wal import apply_wal_with_fallback

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    apply_wal_with_fallback(conn, db_label=db_label)
    if synchronous_full:
        conn.execute("PRAGMA synchronous=FULL")


@contextmanager
def ledger_transaction(
    lock: threading.RLock,
    connect: Callable[[], sqlite3.Connection],
    initialize_schema: Callable[[sqlite3.Connection], None],
) -> Iterator[sqlite3.Connection]:
    """Initialize, transact on, and always close one ledger connection."""
    with lock:
        conn = connect()
        try:
            initialize_schema(conn)
            with conn:
                yield conn
        finally:
            conn.close()
