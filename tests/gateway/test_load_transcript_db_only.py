"""Verify load_transcript returns SQLite messages without any JSONL file."""
from pathlib import Path
import pytest

from gateway.session import SessionStore
from gateway.config import GatewayConfig


def test_load_transcript_returns_db_messages_when_no_jsonl(tmp_path):
    """Reading a transcript must work from SQLite alone — no JSONL fallback needed."""
    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    sid = "test-session-db-only"
    store._db.create_session(session_id=sid, source="test")
    store.append_to_transcript(sid, {"role": "user", "content": "hello", "timestamp": 1.0})
    store.append_to_transcript(sid, {"role": "assistant", "content": "world", "timestamp": 2.0})

    # Delete any JSONL that the current dual-writer left behind
    jsonl_path = tmp_path / f"{sid}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    history = store.load_transcript(sid)
    assert len(history) == 2
    assert history[0]["content"] == "hello"
    assert history[1]["content"] == "world"
