"""The real stop gate releases an owned board run before accepting shutdown."""
from types import SimpleNamespace
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as connections
from agent import turn_stop_gates as gates


def test_turn_boundary_pauses_owned_run_without_injecting_a_nudge(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DRAIN_MARKER", raising=False)
    with connections.connect() as conn:
        task_id = kb.create_task(conn, title="finish current turn", assignee="worker")
        task = kb.claim_task(conn, task_id, claimer="owned-worker")
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
        monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", task.claim_lock)
        kb.request_shutdown_drain()
        monkeypatch.setattr(gates, "_verify_on_stop_nudge", lambda agent: None)
        monkeypatch.setattr(gates, "_pre_verify_nudge", lambda *args: None)
        messages = [{"role": "user", "content": "finish current step"}]
        verdict = gates.apply_stop_gates(SimpleNamespace(), {"role": "assistant", "content": "Step saved."},
            final_response="Step saved.", messages=messages, conversation_history=[],
            pending_verification_response=None, pending_verification_response_previewed=False)
        assert verdict.kanban_shutdown_paused and not verdict.continue_turn
        assert messages == [{"role": "user", "content": "finish current step"}]
        assert kb.get_task(conn, task_id).status == "ready"
        assert kb.latest_run(conn, task_id).outcome == "paused"


def test_opted_in_turn_handoff_uses_real_review_gate(tmp_path, monkeypatch):
    from agent import kanban_stop
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DRAIN_MARKER", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_AUTO_REVIEW_ON_STOP", "1")
    monkeypatch.setattr(kanban_stop, "_configured_review_profile", lambda: "reviewer")
    monkeypatch.setattr(gates, "_verify_on_stop_nudge", lambda agent: None)
    monkeypatch.setattr(gates, "_pre_verify_nudge", lambda *args: None)
    with connections.connect() as conn:
        task_id = kb.create_task(conn, title="review bounded evidence", assignee="worker")
        task = kb.claim_task(conn, task_id, claimer="owned-worker")
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
        monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", task.claim_lock)
        verdict = gates.apply_stop_gates(SimpleNamespace(_kanban_stop_nudges=2),
            {"role": "assistant", "content": "Evidence preserved for independent review."},
            final_response="Evidence preserved for independent review.",
            messages=[], conversation_history=[], pending_verification_response=None,
            pending_verification_response_previewed=False)
        assert not verdict.continue_turn
        reviewed = kb.get_task(conn, task_id)
        assert reviewed.status == "review" and reviewed.assignee == "reviewer"
        assert reviewed.completed_at is None
