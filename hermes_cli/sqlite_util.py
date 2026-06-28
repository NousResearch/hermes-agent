"""Shared SQLite primitives for the small per-profile / board stores.

The projects and kanban stores open WAL SQLite files with the same two
primitives — an idempotent column-add migration and an IMMEDIATE write
transaction. One definition here keeps the two stores from drifting.
"""

from __future__ import annotations

import contextlib
import random
import sqlite3
import time


def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> bool:
    """``ALTER TABLE <table> ADD COLUMN <ddl>``, idempotent across races.

    Returns ``True`` when this call added the column. Swallows the
    ``duplicate column name`` error a concurrent migrator may have run first
    (issue #21708). ``column`` is the human-readable name for the call site;
    ``ddl`` carries the actual definition.
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
        return True
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return False
        raise


_BUSY_MAX_RETRIES = 5
_BUSY_RETRY_MIN_S = 0.020  # 20ms
_BUSY_RETRY_MAX_S = 0.150  # 150ms


def _is_busy_error(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and (
        "database is locked" in str(exc).lower()
        or "database is busy" in str(exc).lower()
    )


def _execute_boundary_with_retry(conn: sqlite3.Connection, sql: str) -> None:
    """Retry BEGIN IMMEDIATE / COMMIT on transient SQLITE_BUSY.

    SQLite's own busy_timeout uses a near-deterministic backoff, so
    concurrent writers re-collide in lockstep under a stampede.  A jittered
    retry on the transaction boundary breaks that convoy.  Only BEGIN
    IMMEDIATE and COMMIT are retried — both are idempotent re-issues that
    touch no transaction body, so a CAS inside write_txn is never replayed.
    """
    for attempt in range(_BUSY_MAX_RETRIES + 1):
        try:
            conn.execute(sql)
            return
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc) or attempt == _BUSY_MAX_RETRIES:
                raise
            time.sleep(random.uniform(_BUSY_RETRY_MIN_S, _BUSY_RETRY_MAX_S))


@contextlib.contextmanager
def write_txn(conn: sqlite3.Connection):
    """An IMMEDIATE write transaction: at most one concurrent writer wins.

    The explicit ROLLBACK is guarded so a SQLite auto-rollback (no active
    transaction left under EIO / lock contention / corruption) cannot shadow
    the original exception with a spurious rollback error.

    BEGIN IMMEDIATE and COMMIT are retried with jitter on transient
    SQLITE_BUSY to break convoy collisions that SQLite's deterministic
    busy_timeout cannot resolve.
    """
    _execute_boundary_with_retry(conn, "BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
    else:
        try:
            _execute_boundary_with_retry(conn, "COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
