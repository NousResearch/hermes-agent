"""The backup admin's in-process state.db opens register with the tracked-connection registry: a reader opened with a bare ``sqlite3.connect`` is a live holder ``has_live_connection()`` cannot see, and raw byte I/O judged safe under it cancels this process's POSIX locks."""

import sqlite3
from pathlib import Path

import pytest

import hermes_cli.sqlite_safe_read as safe_read
from persistence.sqlite.admin._backup import _query_ro_sqlite, _restore_db_pages, _restore_epoch


@pytest.fixture
def state_db(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    db = home / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE runtime_epoch (singleton INTEGER PRIMARY KEY, epoch INTEGER)")
        conn.execute("INSERT INTO sessions VALUES ('s1')")
        conn.execute("INSERT INTO runtime_epoch VALUES (1, 7)")
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


def test_backup_integrity_query_is_tracked(state_db, tracked):
    value, exc = _query_ro_sqlite(state_db, lambda c: c.execute("SELECT count(*) FROM sessions").fetchone()[0])
    assert (value, exc) == (1, None)
    assert _key(state_db) in tracked


def test_restore_destination_connection_is_tracked(state_db, tracked, tmp_path):
    snapshot = tmp_path / "snapshot.db"
    with sqlite3.connect(snapshot) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    conn.close()
    assert _restore_db_pages(snapshot, state_db)
    assert _key(state_db) in tracked


def test_restore_epoch_read_is_tracked(state_db, tracked):
    assert _restore_epoch(state_db) == 7
    assert _key(state_db) in tracked
