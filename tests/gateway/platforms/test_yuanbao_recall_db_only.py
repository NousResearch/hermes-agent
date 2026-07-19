"""Yuanbao recall: branch A1 (exact id) and A2 (content-match) against DB-only transcripts.

state.db persists the platform-side ``message_id`` via the
``platform_message_id`` column (added in the salvage of PR #29211) and
``load_transcript`` surfaces it back on each message dict as ``message_id``
— so the recall guard's exact-id match path stays canonical even with the
JSONL file gone.  When a row has no platform id (e.g. agent-processed
@bot messages whose adapter didn't carry a msg_id, or pre-column legacy
rows), recall falls through to content-match.
"""
from gateway.session import SessionStore
from gateway.config import GatewayConfig


def _pin_db(monkeypatch, tmp_path):
    """Force SessionDB() to write into tmp_path instead of the real ~/.hermes."""
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")


def test_recall_branch_a1_exact_id_match_round_trips_through_db(tmp_path, monkeypatch):
    """A user message persisted with ``message_id`` must round-trip through
    state.db so recall can find and redact it by exact id (branch A1)."""
    _pin_db(monkeypatch, tmp_path)

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    sid = "test-yuanbao-recall-a1"
    store._db.create_session(session_id=sid, source="yuanbao:group:G")
    store.append_to_transcript(sid, {
        "role": "user",
        "content": "sensitive content",
        "timestamp": 1.0,
        "message_id": "platform-msg-abc",
    })
    store.append_to_transcript(sid, {
        "role": "assistant",
        "content": "ack",
        "timestamp": 2.0,
    })

    history = store.load_transcript(sid)
    # The user row must carry its platform id back so the recall guard can
    # match by exact id; the assistant row had no platform id so it should
    # not gain one spuriously.
    user_msg = next(m for m in history if m["role"] == "user")
    assistant_msg = next(m for m in history if m["role"] == "assistant")
    assert user_msg.get("message_id") == "platform-msg-abc"
    assert "message_id" not in assistant_msg

    # Branch A1: locate the row by exact platform id — no content heuristics.
    target = next(
        (m for m in history if m.get("message_id") == "platform-msg-abc"),
        None,
    )
    assert target is not None
    assert target["content"] == "sensitive content"


def test_recall_branch_a2_content_match_when_no_platform_id(tmp_path, monkeypatch):
    """Rows that lack a platform_message_id (e.g. agent-processed @bot
    messages) still match by content as a fallback."""
    _pin_db(monkeypatch, tmp_path)

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    sid = "test-yuanbao-recall-a2"
    store._db.create_session(session_id=sid, source="yuanbao:group:G")
    # No message_id on the dict — simulates an agent-processed message
    # that did not carry the platform msg_id through.
    store.append_to_transcript(sid, {
        "role": "user",
        "content": "sensitive content",
        "timestamp": 1.0,
    })

    history = store.load_transcript(sid)
    assert all("message_id" not in m for m in history)

    # Branch A2: content match recovers the target.
    target = next(
        (m for m in history
         if m.get("role") == "user" and m.get("content") == "sensitive content"),
        None,
    )
    assert target is not None
