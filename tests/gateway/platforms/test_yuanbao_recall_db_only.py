"""Yuanbao recall: branch A2 (content-match) works without JSONL message_id."""
from gateway.session import SessionStore
from gateway.config import GatewayConfig


def test_recall_falls_through_to_content_match_without_message_id(tmp_path, monkeypatch):
    """When transcript has no message_id field, A2 content-match still works.

    Pin DEFAULT_DB_PATH to tmp_path so SessionDB() can't write to the real
    ~/.hermes/state.db. (Module-level constant snapshot, see test_load_transcript_db_only.)
    """
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    sid = "test-yuanbao-recall"
    store._db.create_session(session_id=sid, source="yuanbao:group:G")
    store.append_to_transcript(sid, {"role": "user", "content": "sensitive content", "timestamp": 1.0})
    store.append_to_transcript(sid, {"role": "assistant", "content": "ack", "timestamp": 2.0})

    # The post-PR state: load_transcript returns DB-only, no message_id field.
    history = store.load_transcript(sid)
    assert all("message_id" not in msg for msg in history), \
        "DB-only history should not carry message_id"

    # Branch A2: content match should still find the message
    target = next((m for m in history
                   if m.get("role") == "user" and m.get("content") == "sensitive content"), None)
    assert target is not None
    # Caller would then redact: target["content"] = REDACTED; store.rewrite_transcript(sid, history)
