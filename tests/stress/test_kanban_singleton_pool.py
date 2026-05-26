"""Stress tests for the Kanban singleton connection pool (``get_connection``).

Validates that:

  - ``get_connection`` returns the same connection for repeated calls
  - Different boards get different connections
  - 1000 rapid create+complete operations don't corrupt the DB
  - Concurrent connections maintain integrity_check=ok
  - Pool cleanup runs without errors
"""

from __future__ import annotations

import atexit
import concurrent.futures
import os
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def pool_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a fresh kanban DB via get_connection."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Clear any state from other tests
    kb._INITIALIZED_PATHS.clear()
    kb._connection_pool.clear()
    return home


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------


def test_get_connection_returns_same_object(pool_home):
    """Two calls to get_connection() return the exact same connection."""
    conn1 = kb.get_connection()
    conn2 = kb.get_connection()
    assert conn1 is conn2, "get_connection() must return the same connection"


def test_get_connection_different_boards(pool_home):
    """Different boards get different connections."""
    # Create a second board
    kb.create_board("projx")

    conn_default = kb.get_connection(board="default")
    conn_projx = kb.get_connection(board="projx")

    assert conn_default is not conn_projx, (
        "different boards must have different connections"
    )


def test_connect_is_backward_compatible(pool_home):
    """connect() still works and returns a usable connection."""
    conn = kb.connect()
    # Verify it's a real sqlite3 connection
    assert isinstance(conn, sqlite3.Connection)
    # Verify we can use it
    row = conn.execute("PRAGMA integrity_check").fetchone()
    assert row[0].lower() == "ok"
    conn.close()


def test_pool_connection_is_thread_safe(pool_home):
    """Multiple threads can use get_connection() without issues."""
    conn = kb.get_connection()
    t = kb.create_task(conn, title="thread-safety", assignee="test")

    errors = []
    results = []

    def read_task():
        try:
            c = kb.get_connection()
            task = kb.get_task(c, t)
            results.append(task)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=read_task) for _ in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(errors) == 0, f"unexpected errors: {errors}"
    assert all(r is not None and r.title == "thread-safety" for r in results)


# ---------------------------------------------------------------------------
# Integrity under load
# ---------------------------------------------------------------------------


def test_1000_rapid_operations_no_corruption(pool_home):
    """1000 rapid create + complete operations; integrity_check stays ok."""
    conn = kb.get_connection()

    task_ids = []
    # Create 1000 tasks
    for i in range(1000):
        t = kb.create_task(conn, title=f"load-test-{i}", assignee="test")
        task_ids.append(t)

    # Complete them all
    for t in task_ids:
        kb.complete_task(conn, t, summary="done")

    # Verify integrity
    row = conn.execute("PRAGMA integrity_check").fetchone()
    assert row[0].lower() == "ok", (
        f"integrity_check failed after 1000 ops: {row[0]}"
    )

    # Verify all tasks are done
    for t in task_ids[:10]:  # spot check
        task = kb.get_task(conn, t)
        assert task is not None
        assert task.status == "done"


def test_concurrent_connections_integrity_ok(pool_home):
    """Two connections writing concurrently; integrity_check stays ok."""
    # Use separate connections (not the pool) to simulate concurrent access
    conn1 = kb.connect()
    conn2 = kb.connect()

    errors = []
    done = threading.Event()

    def writer(conn, label):
        try:
            for i in range(100):
                t = kb.create_task(conn, title=f"concurrent-{label}-{i}",
                                   assignee="test")
                kb.complete_task(conn, t, summary="done")
        except Exception as e:
            errors.append((label, e))
        finally:
            done.set()

    t1 = threading.Thread(target=writer, args=(conn1, "A"))
    t2 = threading.Thread(target=writer, args=(conn2, "B"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    conn1.close()
    conn2.close()

    assert len(errors) == 0, f"concurrent writes failed: {errors}"

    # Check integrity on a fresh connection
    check_conn = kb.connect()
    try:
        row = check_conn.execute("PRAGMA integrity_check").fetchone()
        assert row[0].lower() == "ok", (
            f"integrity_check failed after concurrent writes: {row[0]}"
        )
    finally:
        check_conn.close()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_atexit_cleanup_runs_without_errors(pool_home):
    """_cleanup_connections() closes all pooled connections without error."""
    conn = kb.get_connection()
    t = kb.create_task(conn, title="cleanup-test", assignee="test")
    kb.complete_task(conn, t, summary="done")

    # Directly invoke cleanup
    kb._cleanup_connections()

    # Pool should be empty
    assert len(kb._connection_pool) == 0, "pool should be empty after cleanup"

    # Re-acquiring a connection should work (creates a new one)
    conn2 = kb.get_connection()
    assert isinstance(conn2, sqlite3.Connection)
    assert conn2 is not conn, "should be a new connection after cleanup"
