import pytest
import time
from hermes_state import SessionDB
import sqlite3

def test_session_token_events(tmp_path):
    db_path = tmp_path / "test_state.db"
    db = SessionDB(db_path)
    session_id = "test-session-123"
    db.create_session(session_id, "test")

    # Add a token event
    event = {
        "created_at": time.time(),
        "api_call_index": 1,
        "model": "gpt-4",
        "provider": "openai",
        "api_mode": "chat",
        "estimated_input_tokens": 100,
        "actual_input_tokens": 120,
        "output_tokens": 50,
        "breakdown": {"system_prompt": 10, "tool_schemas": 90}
    }
    db.append_token_event(session_id, event)

    events = db.get_token_events(session_id)
    assert len(events) == 1

    e = events[0]
    assert e["api_call_index"] == 1
    assert e["model"] == "gpt-4"
    assert e["estimated_input_tokens"] == 100
    assert e["actual_input_tokens"] == 120
    assert e["output_tokens"] == 50
    # defaults
    assert e["tool_result_tokens"] == 0
    # breakdown json
    assert "breakdown" in e
    assert e["breakdown"]["system_prompt"] == 10

def test_token_events_missing_fields(tmp_path):
    db_path = tmp_path / "test_state.db"
    db = SessionDB(db_path)
    session_id = "test-session-124"
    db.create_session(session_id, "test")

    db.append_token_event(session_id, {"api_call_index": 2})
    events = db.get_token_events(session_id)
    assert len(events) == 1
    assert events[0]["api_call_index"] == 2
    assert events[0]["model"] == ""
    assert events[0]["estimated_input_tokens"] == 0

def test_migration_idempotent(tmp_path):
    db_path = tmp_path / "test_state.db"

    # Initialize SessionDB which should run migrations and create the schema from scratch
    db = SessionDB(db_path)
    session_id = "test-session-125"
    db.create_session(session_id, "test")

    # Check that table exists and works
    db.append_token_event(session_id, {"api_call_index": 1})
    events = db.get_token_events(session_id)
    assert len(events) == 1

    # Re-instantiate to check idempotence
    db2 = SessionDB(db_path)
    events2 = db2.get_token_events(session_id)
    assert len(events2) == 1


def test_delete_session_removes_token_events(tmp_path):
    db_path = tmp_path / "test_state.db"
    db = SessionDB(db_path)
    session_id = "test-session-delete"
    db.create_session(session_id, "test")
    db.append_token_event(session_id, {"api_call_index": 1})

    assert db.delete_session(session_id) is True

    assert db.get_session(session_id) is None
    assert db.get_token_events(session_id) == []


def test_prune_sessions_removes_token_events(tmp_path):
    db_path = tmp_path / "test_state.db"
    db = SessionDB(db_path)
    db.create_session("old", "test")
    db.end_session("old", end_reason="done")
    db.append_token_event("old", {"api_call_index": 1})
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (time.time() - 100 * 86400, "old"),
    )
    db._conn.commit()

    assert db.prune_sessions(older_than_days=90) == 1

    assert db.get_session("old") is None
    assert db.get_token_events("old") == []
