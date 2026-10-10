"""A damaged persisted timestamp must not prevent durable conversation recovery."""
from datetime import datetime

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


@pytest.mark.parametrize("timestamp", [float("inf"), -float("inf"), 1e250, "garbage"])
def test_real_database_recovery_falls_back_for_corrupt_timestamp(tmp_path, monkeypatch, timestamp):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="123")
    try:
        db = store._db
        key = store._generate_session_key(source)
        db.create_session("existing-conversation", "telegram", session_key=key, user_id="123", chat_id="123", chat_type="dm")
        db.append_message("existing-conversation", "user", "remember our project")
        # Model a malformed legacy/imported or repaired cell through real SQLite.
        db._write_sql("UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = ?", (timestamp, timestamp, "existing-conversation"))
        entry = store.get_or_create_session(source)
        assert entry.session_id == "existing-conversation"
        assert entry.created_at == datetime.fromtimestamp(0)
        assert entry.updated_at == entry.created_at
        assert db.get_messages(entry.session_id)[0]["content"] == "remember our project"
    finally:
        store.close_all_db_handles()


def test_valid_durable_timestamps_and_transcript_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="123")
    try:
        db = store._db
        key = store._generate_session_key(source)
        db.create_session("existing-conversation", "telegram", session_key=key, user_id="123", chat_id="123", chat_type="dm")
        db.append_message("existing-conversation", "user", "remember our project")
        started, active = 1700000000.0, 1700003600.0
        db._write_sql("UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = ?", (started, active, "existing-conversation"))
        entry = store.get_or_create_session(source)
        assert entry.session_id == "existing-conversation"
        assert entry.created_at == datetime.fromtimestamp(started)
        assert entry.updated_at == datetime.fromtimestamp(active)
        assert db.get_messages(entry.session_id)[0]["content"] == "remember our project"
    finally:
        store.close_all_db_handles()
