"""Whole-board writer serialization for kanban SQLite databases.

The dispatcher lock from issue #35240 originally guarded only dispatcher ticks.
Every other ``hermes kanban`` process still wrote outside that lock, so a timer
or raw bridge writer could race the gateway on WAL frames. These tests exercise
the shared write-layer lock rather than SQLite's own busy handling.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _hold_dispatch_lock(db_path: str, ready, release) -> None:
    """Child-process holder used to prove the lock is cross-process."""
    with kb._dispatch_tick_lock(Path(db_path)) as held:
        ready.put(held)
        if held:
            release.wait(timeout=15)


def _simple_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "kanban.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE writes(value TEXT NOT NULL)")
    return db_path


def _raw_conn(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path, isolation_level=None)


def test_connection_close_uses_the_board_write_lock(tmp_path, monkeypatch):
    db_path = _simple_db(tmp_path)
    conn = kb._sqlite_connect(db_path)
    locked_paths = []

    @contextlib.contextmanager
    def recording_lock(path, **_kwargs):
        locked_paths.append(Path(path))
        yield

    monkeypatch.setattr(kb, "board_write_lock", recording_lock)
    conn.close()

    assert locked_paths == [db_path.resolve()]


def test_file_length_check_retries_a_transient_checkpoint_snapshot(
    tmp_path,
    monkeypatch,
):
    db_path = _simple_db(tmp_path)
    original_getsize = kb.os.path.getsize
    calls = 0

    def stale_once(path):
        nonlocal calls
        calls += 1
        return 0 if calls == 1 else original_getsize(path)

    conn = kb.connect(db_path=db_path)
    try:
        monkeypatch.setattr(kb.os.path, "getsize", stale_once)
        kb._check_file_length_invariant(conn)
    finally:
        conn.close()

    assert calls == 2


def test_write_txn_fails_closed_while_dispatcher_process_holds_lock(
    tmp_path,
    monkeypatch,
):
    db_path = _simple_db(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Queue()
    release = ctx.Event()
    holder = ctx.Process(
        target=_hold_dispatch_lock,
        args=(str(db_path), ready, release),
    )
    holder.start()
    try:
        assert ready.get(timeout=10) is True
        monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "2000")
        with _raw_conn(db_path) as conn:
            with pytest.raises(
                RuntimeError,
                match="refusing an unsafe concurrent write",
            ):
                with kb.write_txn(conn):
                    conn.execute("INSERT INTO writes VALUES ('unsafe')")
            assert conn.execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 0
    finally:
        release.set()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
    assert holder.exitcode == 0


def test_write_txn_succeeds_after_dispatcher_process_releases_lock(tmp_path):
    db_path = _simple_db(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Queue()
    release = ctx.Event()
    holder = ctx.Process(
        target=_hold_dispatch_lock,
        args=(str(db_path), ready, release),
    )
    holder.start()
    assert ready.get(timeout=10) is True
    release.set()
    holder.join(timeout=10)
    assert holder.exitcode == 0

    with _raw_conn(db_path) as conn:
        with kb.write_txn(conn):
            conn.execute("INSERT INTO writes VALUES ('safe')")
        assert conn.execute("SELECT value FROM writes").fetchall() == [("safe",)]


def test_dispatcher_lock_is_reentrant_for_nested_write_transactions(tmp_path):
    db_path = _simple_db(tmp_path)
    with _raw_conn(db_path) as conn:
        with kb._dispatch_tick_lock(db_path) as held:
            assert held is True
            with kb.write_txn(conn):
                conn.execute("INSERT INTO writes VALUES ('dispatch')")
        assert conn.execute("SELECT value FROM writes").fetchall() == [("dispatch",)]


def test_public_board_write_lock_supports_raw_bridge_writers(tmp_path):
    db_path = _simple_db(tmp_path)
    with kb.board_write_lock(db_path):
        with _raw_conn(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO writes VALUES ('bridge')")
            conn.execute("COMMIT")
            # A core helper nested inside an external integration's lock must
            # reuse the thread-local hold instead of deadlocking itself.
            with kb.write_txn(conn):
                conn.execute("INSERT INTO writes VALUES ('nested')")

    with _raw_conn(db_path) as conn:
        assert conn.execute("SELECT value FROM writes ORDER BY rowid").fetchall() == [
            ("bridge",),
            ("nested",),
        ]


def test_dispatch_uses_the_connection_database_for_lock_scope(tmp_path, monkeypatch):
    selected_db = tmp_path / "selected.db"
    actual_db = tmp_path / "actual.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(selected_db))
    kb.init_db(db_path=actual_db)

    with kb.connect(db_path=actual_db) as conn:
        kb.create_task(conn, title="task", assignee="worker")
        with kb._dispatch_tick_lock(actual_db) as held:
            assert held is True
            result = kb.dispatch_once(conn)

    assert result.skipped_locked is True
    assert selected_db != actual_db
