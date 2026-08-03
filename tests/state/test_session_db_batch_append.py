import sqlite3
from types import SimpleNamespace

import pytest

from hermes_state import CompressionSessionClosedError, SessionDB


def test_append_messages_commits_rows_and_counters_in_one_transaction(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="test", model="m")

    begin_count = {"n": 0}
    original_execute = db._conn.execute

    def counting_execute(sql, *args, **kwargs):
        if isinstance(sql, str) and sql.strip().upper().startswith("BEGIN IMMEDIATE"):
            begin_count["n"] += 1
        return original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(db._conn, "execute", counting_execute)

    written = db.append_messages(
        "s1",
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "checking", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": "ok", "tool_call_id": "c1"},
        ],
    )

    assert written == 3
    assert begin_count["n"] == 1
    session = db._conn.execute(
        "SELECT message_count, tool_call_count FROM sessions WHERE id = ?",
        ("s1",),
    ).fetchone()
    assert dict(session) == {"message_count": 3, "tool_call_count": 1}
    roles = [
        row[0]
        for row in db._conn.execute(
            "SELECT role FROM messages WHERE session_id = ? ORDER BY id",
            ("s1",),
        ).fetchall()
    ]
    assert roles == ["user", "assistant", "tool"]


def test_append_messages_is_atomic_on_locked_retry_exhaustion(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="test", model="m")

    attempts = {"n": 0}
    original_execute = db._conn.execute

    def flaky_execute(sql, *args, **kwargs):
        if isinstance(sql, str) and sql.strip().upper().startswith("BEGIN IMMEDIATE"):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
        return original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(db._conn, "execute", flaky_execute)

    assert db.append_messages("s1", [{"role": "user", "content": "after lock"}]) == 1
    assert attempts["n"] == 2
    count = db._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = 's1'").fetchone()[0]
    assert count == 1


def test_append_messages_retries_transient_compression_busy(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="test", model="m")

    attempts = {"n": 0}
    original_execute = db._conn.execute

    def flaky_execute(sql, *args, **kwargs):
        if isinstance(sql, str) and sql.startswith("SELECT holder FROM compression_locks"):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return SimpleNamespace(fetchone=lambda: {"holder": "other"})
        return original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(db._conn, "execute", flaky_execute)
    monkeypatch.setattr("hermes_state.time.sleep", lambda _seconds: None)

    assert db.append_messages("s1", [{"role": "user", "content": "after compression"}]) == 1
    assert attempts["n"] == 2


def test_append_messages_respects_compression_closed_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="test", model="m")
    db._conn.execute(
        "UPDATE sessions SET ended_at = 1, end_reason = 'compression' WHERE id = 's1'"
    )

    with pytest.raises(CompressionSessionClosedError):
        db.append_messages("s1", [{"role": "user", "content": "blocked"}])

    count = db._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = 's1'").fetchone()[0]
    assert count == 0
