"""Review receipts are display history, never new model instructions."""
import json
from types import SimpleNamespace

import pytest

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


def test_idle_queue_freezes_origin_before_new_and_preserves_it_through_requeue(tmp_path, monkeypatch):
    from agent.review_idle_queue import ReviewIdleQueue
    import pytest

    db = SessionDB(tmp_path / "queued.db")
    for sid in ("origin", "next-chat"):
        db.create_session(sid, source="desktop")
    agent = parent(db, "origin")
    original_events, successor_events = [], []
    agent.background_review_event_callback = original_events.append
    queue = ReviewIdleQueue()
    monkeypatch.setattr(queue, "_ensure_thread", lambda: None)
    monkeypatch.setattr(queue, "_still_enabled", lambda item: True)
    clock = [0.0]
    queue._now = lambda: clock[0]
    queue._server_idle = lambda: True
    queue._quiet_since = -1000.0
    calls = []

    def finish_fork(agent, snapshot, prompt, config, run, state, review_memory, explicit):
        state.review_messages = [
            {"role": "assistant", "tool_calls": [{"id": "queued-write", "type": "function", "function": {
                "name": "skill_manage", "arguments": json.dumps({"action": "patch", "name": "queued"})}}]},
            {"role": "tool", "tool_call_id": "queued-write", "content": json.dumps({
                "success": True, "message": "Skill 'queued' patched"})},
        ]
    monkeypatch.setattr(review, "_run_review_fork", finish_fork)

    def dispatch(**kwargs):
        run = review.prepare_background_review_run(agent)
        assert run is not None
        calls.append(run.source_session_id)
        try:
            if len(calls) == 1:
                # A foreground preemption requeues while the worker still carries
                # origin's Context; /new has already moved the reusable parent.
                queue.enqueue(agent, "origin-key", kwargs)
            else:
                review._run_review_in_thread(agent, [], "Review", review_run=run)
        finally:
            review.finish_background_review_run(agent, run)
            if hasattr(run, "worker_done"):
                from agent.review_lifecycle import finish_review_worker
                finish_review_worker(agent, run)
    agent._spawn_background_review_now = dispatch
    queue.enqueue(agent, "origin-key", {"task_cfg": {}})
    agent.session_id = "next-chat"
    agent.background_review_event_callback = successor_events.append
    clock[0] = 1000.0

    class Done(BaseException):
        pass
    def wake():
        if len(calls) >= 2:
            raise Done
    monkeypatch.setattr(queue._wake, "wait", wake)
    with pytest.raises(Done):
        queue._run()
    assert calls == ["origin", "origin"]
    assert len(receipts(db, "origin")) == 1
    assert receipts(db, "next-chat") == []
    assert len(original_events) == 1
    assert original_events[0]["stored_session_id"] == "origin"
    assert successor_events == []
    db.close()


@pytest.mark.parametrize("channel", ["structured", "text", "absent"])
def test_delayed_review_postlude_keeps_own_delivery_after_successor_starts(tmp_path, monkeypatch, channel):
    db = SessionDB(tmp_path / "postlude.db")
    for sid in ("origin", "successor"):
        db.create_session(sid, source="desktop")
    agent = parent(db, "origin")
    original_events, original_texts, successor_events, successor_texts = [], [], [], []
    if channel == "structured":
        agent.background_review_event_callback = original_events.append
    if channel != "absent":
        agent.background_review_callback = original_texts.append
    run = review.prepare_background_review_run(agent)
    assert run is not None
    successor_runs = []

    def finish_fork(agent, snapshot, prompt, config, run, state, review_memory, explicit):
        state.review_messages = [
            {"role": "assistant", "tool_calls": [{"id": "write", "type": "function", "function": {
                "name": "skill_manage", "arguments": json.dumps({"action": "patch", "name": "canary"})}}]},
            {"role": "tool", "tool_call_id": "write", "content": json.dumps({
                "success": True, "message": "Skill 'canary' patched"})},
        ]
        # Request completion permits the successor turn while this worker still
        # has to summarize and deliver the old review's completed writes.
        review.finish_background_review_run(agent, run)
        assert run.request_done.is_set()
        agent.session_id = "successor"
        agent.background_review_event_callback = successor_events.append
        agent.background_review_callback = successor_texts.append
        successor_runs.append(review.prepare_background_review_run(agent))

    monkeypatch.setattr(review, "_run_review_fork", finish_fork)
    try:
        review._run_review_in_thread(agent, [], "Review", review_run=run)
        assert successor_events == successor_texts == []
        assert len(original_events) == (1 if channel == "structured" else 0)
        assert len(original_texts) == (1 if channel == "text" else 0)
        if original_events:
            assert original_events[0]["stored_session_id"] == "origin"
            assert original_events[0]["persisted"] is True
        assert len(receipts(db, "origin")) == 1
        assert receipts(db, "successor") == []
        assert successor_runs[0] is not None
        assert agent._background_review_run is successor_runs[0]
    finally:
        for successor in successor_runs:
            review.finish_background_review_run(agent, successor)
        db.close()
