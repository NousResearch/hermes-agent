"""The store admin's and the readiness probe's in-process state.db opens register with the tracked-connection
registry: a reader opened with a bare ``sqlite3.connect`` is a live holder ``has_live_connection()`` cannot see, and
raw byte I/O judged safe under it cancels this process's POSIX locks."""

import sqlite3
from pathlib import Path

import pytest

import hermes_cli.sqlite_safe_read as safe_read
import persistence.sqlite.admin._doctor as doctor
from persistence.sqlite.admin import SQLiteStoreAdmin
from persistence.sqlite.services import probe_state_db


@pytest.fixture
def state_db(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    db = home / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("CREATE TABLE session_admissions (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE worker_executions (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO sessions VALUES ('s1')")
        conn.execute("INSERT INTO messages (content) VALUES ('hello')")
        conn.execute("INSERT INTO session_admissions VALUES ('a1')")
    conn.close()
    return db


@pytest.fixture
def tracked(monkeypatch):
    """Keys registered as live while the code under test ran."""
    seen: list[str] = []
    real = safe_read._track_key

    def _spy(key, delta=1):
        if delta > 0:
            seen.append(key)
        return real(key, delta)

    monkeypatch.setattr(safe_read, "_track_key", _spy)
    return seen


def _key(path: Path) -> str:
    return str(path.resolve())


def test_session_row_counts_read_is_tracked(state_db, tracked):
    assert SQLiteStoreAdmin(state_db).session_row_counts() == (1, 1)
    assert _key(state_db) in tracked


def test_runtime_ledger_rows_probe_is_tracked(state_db, tracked):
    assert SQLiteStoreAdmin.has_runtime_ledger_rows(state_db) is True
    assert _key(state_db) in tracked


def test_doctor_session_row_count_is_tracked(state_db, tracked):
    assert SQLiteStoreAdmin(state_db).session_row_count() == 1
    assert _key(state_db) in tracked


def test_write_health_snapshot_source_is_tracked(state_db, tracked, monkeypatch):
    monkeypatch.setattr(doctor, "_live_writer_holds_db", lambda _path: True)
    SQLiteStoreAdmin(state_db).write_health_reason(should_fix=True)
    assert _key(state_db) in tracked


def test_integrity_check_and_journal_mode_are_tracked(state_db, tracked):
    admin = SQLiteStoreAdmin(state_db)
    assert admin.integrity_check() == "ok"
    assert admin.journal_mode() == "wal"
    assert tracked.count(_key(state_db)) >= 2


def test_set_journal_mode_switch_connection_is_tracked(state_db, tracked):
    assert SQLiteStoreAdmin(state_db, read_only=False).set_journal_mode("delete", force=True) == ("wal", "delete")
    assert _key(state_db) in tracked


def test_readiness_probe_is_tracked(state_db, tracked):
    assert probe_state_db(state_db).status == "ok"
    assert _key(state_db) in tracked


def test_readiness_probe_opens_the_tracked_file_with_reserved_home_characters(
    state_db, tmp_path, monkeypatch, tracked,
):
    """URI parsing must not redirect a tracked readiness open to a different file."""
    home = tmp_path / "profile # with spaces"  # no "?": Windows rejects it in a file name
    state_db.parent.rename(home)
    db_path = home / "state.db"
    opened_paths: list[str] = []
    real_connect = safe_read.connect_tracked

    def _observe_open(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        row = conn.execute("PRAGMA database_list").fetchone()
        opened_paths.append(_key(Path(row[2])))
        return conn

    monkeypatch.setattr(safe_read, "connect_tracked", _observe_open)
    assert probe_state_db(db_path).status == "ok"
    assert opened_paths == [_key(db_path)]
    assert _key(db_path) in tracked
