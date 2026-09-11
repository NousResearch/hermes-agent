"""Regression coverage for bounded session-list SQLite lock retries (ENG-1265)."""

import logging
import sqlite3

import pytest

import hermes_state_sessions
from hermes_state import SessionDB


def test_session_list_retry_is_bounded_and_rejects_unrelated_errors(
    tmp_path, monkeypatch, caplog
):
    db_path = tmp_path / "state.db"
    writable = SessionDB(db_path=db_path)
    writable.create_session(session_id="s1", source="cli", model="m")
    writable.close()

    mode_setter = sqlite3.connect(db_path)
    assert mode_setter.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    mode_setter.close()

    reader = SessionDB(db_path=db_path, read_only=True)
    assert reader._conn is not None
    reader._conn.execute("PRAGMA busy_timeout=0")
    holder = sqlite3.connect(db_path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    sleeps = []
    monkeypatch.setattr(hermes_state_sessions.time, "sleep", sleeps.append)
    try:
        with caplog.at_level(logging.INFO, logger="hermes_state"):
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                reader.list_sessions_rich(limit=10)
        assert len(sleeps) == hermes_state_sessions._SESSION_LIST_LOCK_RETRY_ATTEMPTS
        messages = [record.getMessage() for record in caplog.records]
        assert sum("contention retry" in message for message in messages) == len(sleeps)
        exhausted = next(
            message for message in messages if "contention exhausted" in message
        )
        assert "phase=page" in exhausted
        assert f"retry_count={len(sleeps)}" in exhausted
        assert "connection_mode=read_only_primary" in exhausted
        assert "fallback=False" in exhausted
    finally:
        holder.rollback()
        holder.close()

    assert hermes_state_sessions._is_sqlite_lock_contention(
        sqlite3.OperationalError("database schema is locked: main")
    )
    assert not hermes_state_sessions._is_sqlite_lock_contention(
        sqlite3.OperationalError("resource is busy")
    )

    destroyer = sqlite3.connect(reader.db_path)
    destroyer.execute("DROP TABLE sessions")
    destroyer.commit()
    destroyer.close()
    sleep_count = len(sleeps)
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such table: sessions"):
            reader.list_sessions_rich(limit=10)
        assert len(sleeps) == sleep_count
    finally:
        reader.close()
