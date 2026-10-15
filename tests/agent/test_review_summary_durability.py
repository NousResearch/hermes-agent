"""Review receipts are display history, never new model instructions."""
import json
from types import SimpleNamespace

from agent import background_review as review
from hermes_state import SessionDB


def parent(db, session_id):
    return SimpleNamespace(
        _session_db=db, session_id=session_id, _persist_disabled=False,
        _safe_print=lambda text: None, background_review_callback=None,
        background_review_event_callback=None, memory_notifications="on",
        _emit_auxiliary_failure=lambda *args: None,
    )


def receipts(db, session_id):
    return [m for m in db.get_resume_conversations(session_id)[1]
            if m.get("display_kind") == "review_summary"]


def test_confirmation_survives_reopen_and_compaction_without_changing_model_history(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("chat", source="desktop")
    db.append_message("chat", "user", "Inspect this procedure")
    db.append_message("chat", "assistant", "Done")
    before = db.get_resume_conversations("chat")[0]
    agent = parent(db, "chat")
    events = []
    agent.background_review_event_callback = events.append
    review._publish_review_summary(agent, ["Skill 'canary' patched"])
    rows = receipts(db, "chat")
    assert len(rows) == 1, "the visible confirmation must be durable before delivery"
    assert len(events) == 1
    receipt = rows[0]
    assert events[0]["review_id"] == receipt["display_metadata"]["review_id"]
    assert events[0]["row_id"] == receipt["_row_id"]
    assert events[0]["timestamp"] == receipt["timestamp"]
    assert db.get_resume_conversations("chat")[0] == before
    assert all(m.get("display_kind") != "review_summary"
               for m in db.get_messages_as_conversation("chat", repair_alternation=True))
    db.close()
    db = SessionDB(path)
    assert receipts(db, "chat") == rows
    # A compacted model tail is shorter than the display history; the receipt
    # cannot consume a verbatim-tail slot or be mistaken for model context.
    db.archive_and_compact("chat", [{"role": "assistant", "content": "Done"}], tail_count=1)
    assert len(receipts(db, "chat")) == 1
    assert all(m.get("display_kind") != "review_summary"
               for m in db.get_resume_conversations("chat")[0])
    db.close()


def test_completed_review_stays_with_its_origin_when_parent_moves_to_another_session(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    for sid in ("origin", "next-chat"):
        db.create_session(sid, source="desktop")
    agent = parent(db, "origin")
    events = []
    agent.background_review_event_callback = events.append

    def finish_fork(agent, snapshot, prompt, config, run, state, review_memory, explicit):
        agent.session_id = "next-chat"
        state.review_messages = [
            {"role": "assistant", "tool_calls": [{"id": "write-1", "type": "function", "function": {
                "name": "skill_manage", "arguments": json.dumps({"action": "patch", "name": "canary"})}}]},
            {"role": "tool", "tool_call_id": "write-1", "content": json.dumps({
                "success": True, "message": "Skill 'canary' patched"})},
        ]

    monkeypatch.setattr(review, "_run_review_fork", finish_fork)
    review._run_review_in_thread(agent, [], "Review")
    assert len(receipts(db, "origin")) == 1
    assert receipts(db, "next-chat") == []
    assert events[0]["stored_session_id"] == "origin"
    # Another profile with an identically named session sees none of the receipt.
    other = SessionDB(tmp_path / "other-profile.db")
    other.create_session("origin", source="desktop")
    assert receipts(other, "origin") == []
    other.close()
    db.close()
