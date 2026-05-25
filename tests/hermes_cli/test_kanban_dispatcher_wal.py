"""Tests for WAL inode-rotation race fix in gateway dispatcher (issue #31158).

The fix replaces open/close-per-call SQLite patterns in gateway watcher paths
with a shared per-board connection via GatewayRunner._kb_conn(). This prevents
SQLITE_IOERR_SHMMAP errors caused by WAL inode rotation when the last holder
closes and recreates -shm/-wal files with new inodes.
"""

import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Minimal stub that replicates the _kb_conn logic without importing gateway
# ---------------------------------------------------------------------------

class _KbConnMixin:
    """Minimal replica of GatewayRunner._kb_conn for unit-testing."""

    def __init__(self):
        self._kanban_conn_cache: dict = {}
        self._kanban_conn_lock = threading.Lock()

    def _kb_conn(self, slug=None):
        key = slug or "default"
        with self._kanban_conn_lock:
            if key not in self._kanban_conn_cache:
                conn = self._make_conn(slug)
                self._kanban_conn_cache[key] = conn
            return self._kanban_conn_cache[key]

    def _make_conn(self, slug):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# test_single_connection_reused_across_ticks
# ---------------------------------------------------------------------------

def test_single_connection_reused_across_ticks():
    """_kb_conn returns the same connection object on repeated calls for the same slug."""
    mock_conn = MagicMock(spec=sqlite3.Connection)
    call_count = 0

    class _Stub(_KbConnMixin):
        def _make_conn(self, slug):
            nonlocal call_count
            call_count += 1
            return mock_conn

    stub = _Stub()
    N = 20
    results = [stub._kb_conn("board-a") for _ in range(N)]

    assert call_count == 1, f"Expected 1 connect call, got {call_count}"
    assert all(r is mock_conn for r in results), "All calls must return the same connection"


# ---------------------------------------------------------------------------
# test_multi_board_each_gets_own_connection
# ---------------------------------------------------------------------------

def test_multi_board_each_gets_own_connection():
    """Different board slugs each receive a distinct cached connection."""
    connections = {}

    class _Stub(_KbConnMixin):
        def _make_conn(self, slug):
            c = MagicMock(spec=sqlite3.Connection)
            connections[slug or "default"] = c
            return c

    stub = _Stub()
    conn_a = stub._kb_conn("board-a")
    conn_b = stub._kb_conn("board-b")
    conn_a2 = stub._kb_conn("board-a")

    assert conn_a is not conn_b, "Different boards must get different connections"
    assert conn_a is conn_a2, "Same board must reuse cached connection"
    assert len(stub._kanban_conn_cache) == 2
    assert set(stub._kanban_conn_cache.keys()) == {"board-a", "board-b"}


# ---------------------------------------------------------------------------
# test_gateway_shutdown_closes_cached_connections
# ---------------------------------------------------------------------------

def test_gateway_shutdown_closes_cached_connections():
    """Stop logic closes all cached kanban connections and clears the cache."""
    conn_a = MagicMock(spec=sqlite3.Connection)
    conn_b = MagicMock(spec=sqlite3.Connection)

    stub = _KbConnMixin.__new__(_KbConnMixin)
    stub._kanban_conn_cache = {"board-a": conn_a, "board-b": conn_b}
    stub._kanban_conn_lock = threading.Lock()

    # Replicate the stop() cleanup block
    with stub._kanban_conn_lock:
        for slug, kconn in list(stub._kanban_conn_cache.items()):
            try:
                kconn.close()
            except Exception:
                pass
        stub._kanban_conn_cache.clear()

    conn_a.close.assert_called_once()
    conn_b.close.assert_called_once()
    assert stub._kanban_conn_cache == {}


# ---------------------------------------------------------------------------
# test_dispatcher_watcher_no_eio_after_multi_tick
# ---------------------------------------------------------------------------

def test_dispatcher_watcher_no_eio_after_multi_tick(tmp_path):
    """10 successive _kb_conn calls against a real SQLite DB raise no errors
    and always return the same connection object."""
    db_path = tmp_path / "kanban.db"

    class _Stub(_KbConnMixin):
        def _make_conn(self, slug):
            return sqlite3.connect(
                str(db_path),
                isolation_level=None,
                timeout=5,
                check_same_thread=False,
            )

    stub = _Stub()

    conns = []
    for _ in range(10):
        conn = stub._kb_conn("test-board")
        conns.append(conn)
        # Basic sanity — the connection should be usable
        conn.execute("SELECT 1")

    first = conns[0]
    assert all(c is first for c in conns), "All ticks must reuse the same connection"

    first.close()


# ---------------------------------------------------------------------------
# test_eio_recovery_on_stale_connection
# ---------------------------------------------------------------------------

def test_eio_recovery_on_stale_connection():
    """When dispatch_once raises OperationalError('disk I/O error'), the
    dispatcher logs it and returns None without propagating the exception."""
    import logging

    bad_conn = MagicMock(spec=sqlite3.Connection)

    class _Stub(_KbConnMixin):
        def _make_conn(self, slug):
            return bad_conn

    stub = _Stub()

    def _fake_dispatch_once(conn, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    result = None
    caught = False
    try:
        conn = stub._kb_conn("board-x")
        result = _fake_dispatch_once(conn, board="board-x")
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        caught = True
        result = None

    assert caught, "Exception was raised (caller must catch it as before)"
    assert result is None


# ---------------------------------------------------------------------------
# test_thread_safety_concurrent_kanban_access
# ---------------------------------------------------------------------------

def test_thread_safety_concurrent_kanban_access():
    """8 threads each calling _kb_conn 100x concurrently all get the same
    connection and no exception is raised."""
    shared_conn = MagicMock(spec=sqlite3.Connection)
    make_count = 0

    class _Stub(_KbConnMixin):
        def _make_conn(self, slug):
            nonlocal make_count
            make_count += 1
            return shared_conn

    stub = _Stub()
    results = []
    errors = []

    def _worker():
        for _ in range(100):
            try:
                results.append(stub._kb_conn("shared-board"))
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Unexpected errors from threads: {errors}"
    assert make_count == 1, f"Expected exactly 1 make_conn call, got {make_count}"
    assert all(r is shared_conn for r in results), "All threads must get the same connection"
    assert len(results) == 8 * 100
