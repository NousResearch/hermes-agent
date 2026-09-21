#!/usr/bin/env python3
"""Cross-process write lock for kanban DBs.

Standalone scripts (kanban-reconcile, triage-processor, skill-reroute,
quality-gate, etc.) must use this instead of plain sqlite3.connect()+commit()
to prevent the WAL checkpoint race that corrupts index pages under concurrent
write load across processes.

Usage:

    from kanban_write_lock import write_lock, open_kanban_db, unblock_task

    # Option A: context manager (recommended)
    with open_kanban_db("/path/to/kanban.db") as conn:
        with write_lock(conn):
            conn.execute("UPDATE tasks SET ...")
            conn.commit()

    # Option B: standalone lock (for raw connections)
    conn = sqlite3.connect("/path/to/kanban.db")
    with write_lock(conn):
        conn.execute("UPDATE tasks SET ...")
        conn.commit()
    conn.close()

    # Option C: unblock_task (one-call complete unblock)
    with open_kanban_db("/path/to/kanban.db") as conn:
        with write_lock(conn):
            unblock_task(conn, "t_xxx", "Unblocked by KENSEI: fixed root cause")

The lock uses fcntl.flock(LOCK_EX) on a .write_lock sidecar file next to the
database. Only one process may hold the lock at any instant, eliminating the
WAL checkpoint race.

Python's sqlite3 module auto-uses DEFERRED transactions by default.
write_lock() does NOT auto-commit — callers must call conn.commit() explicitly
inside the lock context. This preserves rollback-on-exception semantics.
For multi-statement atomicity, do:
    with write_lock(conn):
        conn.execute("BEGIN IMMEDIATE")
        ...
        conn.commit()
"""

import contextlib
import fcntl
import json
import os
import sqlite3
import time
from pathlib import Path


def _resolve_lock_path(conn: sqlite3.Connection) -> Path | None:
    """Resolve the .write_lock sidecar path for a connection.

    Uses PRAGMA database_list to get the main DB path on disk.
    Returns None if the path cannot be resolved (e.g. :memory: or URI mode).
    """
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
        if row is None:
            return None
        db_path = row[2]
    except Exception:
        return None
    if not db_path or db_path == ":memory:":
        return None
    # Handle URI-mode connections: extract the path from "file:/path?params"
    if db_path.startswith("file:"):
        # Strip "file:" prefix and any query params
        clean = db_path[5:]
        if "?" in clean:
            clean = clean.split("?")[0]
        if not clean:
            return None
        db_path = clean
    p = Path(db_path)
    return p.with_name(p.name + ".write_lock")


@contextlib.contextmanager
def write_lock(conn: sqlite3.Connection):
    """Cross-process write lock for a kanban DB connection.

    Acquires fcntl.flock(LOCK_EX) on the .write_lock sidecar file. Blocks
    until the lock is available. Releases on exit.

    Does NOT auto-commit. Callers must call conn.commit() inside the context.
    On exception, the transaction is rolled back (not committed).
    """
    lock_path = _resolve_lock_path(conn)
    if lock_path is None:
        # Can't resolve path — yield without locking.
        # Caller should check and handle this case if needed.
        yield conn
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield conn
        except Exception:
            # Rollback on exception, don't commit partial work
            try:
                conn.rollback()
            except Exception:
                pass
            raise
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


@contextlib.contextmanager
def open_kanban_db(path: str | Path, timeout: int = 5):
    """Open a kanban DB with write access and a busy timeout.

    Yields a writable sqlite3.Connection. The connection is closed on exit.
    Does NOT auto-commit — callers must commit explicitly.
    On exception, the transaction is rolled back.

    Usage:

        with open_kanban_db("/path/to/kanban.db") as conn:
            with write_lock(conn):
                conn.execute("UPDATE ...")
                conn.commit()
    """
    conn = sqlite3.connect(str(path), timeout=timeout)
    try:
        yield conn
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def unblock_task(conn: sqlite3.Connection, task_id: str, reason: str) -> bool:
    """Complete unblock: set status=todo, clear stale fields, emit unblocked event.

    This is the ONLY safe way to unblock a task. It:
    - Sets status='todo'
    - Clears claim_lock, claim_expires, worker_pid, started_at, current_run_id
    - Resets consecutive_failures=0 and last_failure_error=NULL
    - Emits an 'unblocked' task_event so the sticky-block check passes
    - Commits the transaction

    Must be called inside a write_lock(conn) context.

    Returns True if the task was found and updated, False otherwise.
    """
    if not task_id or not isinstance(task_id, str):
        return False

    now = int(time.time())
    cur = conn.execute(
        """UPDATE tasks SET
           status='todo',
           status_reason=?,
           claim_lock=NULL,
           claim_expires=NULL,
           worker_pid=NULL,
           started_at=NULL,
           consecutive_failures=0,
           last_failure_error=NULL,
           current_run_id=NULL,
           updated_at=?
        WHERE id=?""",
        (reason[:500], now, task_id),
    )
    if cur.rowcount == 0:
        return False

    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, 'unblocked', ?, ?)",
        (task_id, json.dumps({"reason": reason, "source": "kanban_write_lock.unblock_task"}), now),
    )
    conn.commit()
    return True