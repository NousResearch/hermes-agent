"""A muted wake keeps durable evidence while hiding only its new presentation rows."""
from types import SimpleNamespace

from agent.notification_presentation import notification_turn
from agent.session_persistence import _db_flush_collect
from hermes_state import SessionDB
from tui_gateway.server import _history_to_messages


def test_muted_wake_preserves_history_and_new_evidence(tmp_path):
    old = {"role": "user", "content": "human request"}
    diagnostic = {"role": "user", "content": "worker failed"}
    reply = {"role": "assistant", "content": "model diagnostic echo"}
    messages = [old, diagnostic, reply]
    agent = SimpleNamespace(session_id="session", _last_flushed_db_idx=0)
    with notification_turn(agent, muted=True):
        rows, _ = _db_flush_collect(agent, messages, [old])
    assert "display_kind" not in old
    assert [row["content"] for row in rows] == ["worker failed", "model diagnostic echo"]
    assert all(row["display_kind"] == "hidden" for row in rows)
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("session", source="tui")
        db.append_messages_batch("session", rows)
        stored = db.get_messages("session")
    assert [row["content"] for row in stored] == ["worker failed", "model diagnostic echo"]
    assert _history_to_messages(stored) == []
    next_result = {"role": "assistant", "content": "requested result"}
    rows, _ = _db_flush_collect(agent, [next_result], [])
    assert rows[0]["display_kind"] is None
