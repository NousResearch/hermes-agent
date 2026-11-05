"""Verify load_transcript returns SQLite messages without any JSONL file."""


from gateway.session import SessionStore
from gateway.config import GatewayConfig
from tests.fixtures.storage_seed import seed_session


def test_load_transcript_returns_db_messages_when_no_jsonl(tmp_path, monkeypatch):
    """Reading a transcript must work from SQLite alone — no JSONL fallback needed."""
    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    sid = "test-session-db-only"
    seed_session(store.persistence, sid, "test")
    store.append_to_transcript(sid, {"role": "user", "content": "hello", "timestamp": 1.0})
    store.append_to_transcript(sid, {"role": "assistant", "content": "world", "timestamp": 2.0})

    history = store.load_transcript(sid)
    assert len(history) == 2
    assert history[0]["content"] == "hello"
    assert history[1]["content"] == "world"
