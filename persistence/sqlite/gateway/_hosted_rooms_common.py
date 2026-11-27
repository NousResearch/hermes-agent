"""The SQL helpers the hosted-room repositories share inside their transactions, and the epoch fence of the owner verbs, which refuses a stale epoch as the contract's ``FenceStale``."""

from __future__ import annotations

import sqlite3

from persistence import FenceStale
from persistence.core.errors import StaleFenceKind

__all__ = ["_epoch", "fenced_update", "table_columns", "table_exists"]


def fenced_update(conn: sqlite3.Connection, sql: str, params: tuple, error: Exception) -> None:
    """Run a compare-and-swap UPDATE; anything but exactly one row means the fence was lost."""
    if conn.execute(sql, params).rowcount != 1:
        raise error


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    return frozenset(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))


def _epoch(conn: sqlite3.Connection, epoch: int) -> None:
    row = conn.execute('SELECT epoch FROM runtime_epoch WHERE singleton=1').fetchone()
    if type(epoch) is not int or row is None or row[0] != epoch:
        raise FenceStale(StaleFenceKind.STALE_EPOCH, "runtime_epoch", "singleton")
