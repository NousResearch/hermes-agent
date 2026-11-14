"""Directory refresh must not materialize stored prompts (#139265)."""

import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.channel_directory import _build_from_sessions_db
from hermes_state import SessionDB


def test_directory_reads_routing_without_prompt_payloads(tmp_path, monkeypatch):
    import hermes_state_registry

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(
            "contact", "discord", session_key="agent:main:discord:dm:42",
            chat_id="42", display_name="Contact", system_prompt="large prompt " * 10000,
        )
        monkeypatch.setattr(hermes_state_registry, "acquire", lambda: db)
        monkeypatch.setattr(hermes_state_registry, "release_or_close", lambda acquired: None)
        reads = []

        def observe(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_READ:
                reads.append((table, column))
            return sqlite3.SQLITE_OK

        read_ctx = db._read_ctx

        @contextmanager
        def observed_read():
            with read_ctx() as conn:
                conn.set_authorizer(observe)
                try:
                    yield conn
                finally:
                    conn.set_authorizer(None)

        monkeypatch.setattr(db, "_read_ctx", observed_read)
        entries = _build_from_sessions_db("discord")
        assert entries == [{"id": "42", "name": "Contact", "type": "dm", "thread_id": None}]
        assert not ({("sessions", "system_prompt"), ("system_prompts", "prompt")} & set(reads)), reads
    finally:
        db.close()


@pytest.mark.parametrize("platform", [None, "DISCORD", "telegram", "missing"])
def test_origin_projection_preserves_full_reader_rows_and_order(tmp_path, monkeypatch, platform):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        for sid, source, key, started, active in (
            ("old", "discord", "d", 1, 100),
            ("new", "discord", "d", 2, 4),
            ("thread", "discord", "t", 3, 8),
            ("telegram", "telegram", "g", 4, 6),
            ("unkeyed", "discord", None, 5, 20),
        ):
            db.create_session(sid, source, session_key=key, chat_id=sid, display_name=sid,
                              thread_id="topic" if sid == "thread" else None)
            db._write_sql("UPDATE sessions SET started_at=?,last_activity_at=? WHERE id=?",
                          (started, active, sid))
        db.end_session("new", "session_switch")
        full = db.list_gateway_sessions(platform=platform, active_only=False)
        flush = Mock(side_effect=AssertionError("directory reads must not flush accounting"))
        monkeypatch.setattr(db, "flush_token_counts", flush)
        rows = db.list_gateway_session_origins(platform=platform)
        keys = ("origin_json", "chat_id", "thread_id", "display_name", "chat_type", "last_active")
        assert rows == [{key: row[key] for key in keys} for row in full]
        flush.assert_not_called()
        monkeypatch.undo()
    finally:
        db.close()


@pytest.mark.parametrize("kind", ["legacy", "empty", "error", "absent"])
def test_reader_compatibility_release_and_json_fallback(tmp_path, monkeypatch, kind):
    import json
    import hermes_state_registry
    from gateway.channel_directory import _build_from_sessions

    row = {"chat_id": "42", "display_name": "Contact"}
    db = SimpleNamespace()
    if kind == "legacy":
        db.list_gateway_sessions = Mock(return_value=[row])
    elif kind == "empty":
        db.list_gateway_session_origins = Mock(return_value=[])
    elif kind == "error":
        db.list_gateway_session_origins = Mock(side_effect=RuntimeError("read failed"))
    release = Mock()
    monkeypatch.setattr(hermes_state_registry, "acquire", lambda: db)
    monkeypatch.setattr(hermes_state_registry, "release_or_close", release)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "sessions.json").write_text(json.dumps({
        "legacy": {"origin": {"platform": "discord", "chat_id": "42", "chat_name": "Contact"}}
    }), encoding="utf-8")
    assert _build_from_sessions("discord") == [
        {"id": "42", "name": "Contact", "type": "dm", "thread_id": None}]
    release.assert_called_once_with(db)
    if kind == "legacy":
        db.list_gateway_sessions.assert_called_once_with(platform="discord", active_only=False)
