"""Yuanbao recall: branch A (content-match) works against DB-only transcripts."""
from gateway.session import SessionStore
from gateway.config import GatewayConfig


def test_recall_content_match_finds_target_in_db_transcript(tmp_path, monkeypatch):
    """state.db doesn't preserve message_id, so recall uses content-match.

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

    # DB-only history carries no platform message_id (PR #29211 dropped that path).
    history = store.load_transcript(sid)
    assert all("message_id" not in msg for msg in history)

    # Branch A: content match finds the target row that recall would redact.
    target = next((m for m in history
                   if m.get("role") == "user" and m.get("content") == "sensitive content"), None)
    assert target is not None
    # Caller would then redact: target["content"] = REDACTED; store.rewrite_transcript(sid, history)
