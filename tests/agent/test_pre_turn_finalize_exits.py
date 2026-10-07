"""Terminal handoffs and early-return history regressions for finalize hooks."""
import json
from unittest.mock import MagicMock, patch

import pytest

from tests.agent.test_pre_turn_finalize_hook import _e2e_agent, _e2e_response


def _hook(monkeypatch, enabled=True):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: enabled and name == "pre_turn_finalize")
    hook = MagicMock(return_value="internal plugin continuation")
    monkeypatch.setattr("hermes_cli.plugins.get_pre_turn_finalize_continue_message", hook)
    return hook


def _board(tmp_path, monkeypatch, review=False):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    conn = kbc.connect()
    tid = kb.create_task(conn, title="test task", assignee="builder")
    task = kb.claim_task(conn, tid)
    assert task is not None
    if review:
        assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer", expected_run_id=task.current_run_id)
        task = kb.claim_review_task(conn, tid)
        assert task is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", task.claim_lock)
    return kb, conn, tid, task.current_run_id


@pytest.mark.parametrize("tool,outcome", [("complete", "completed"), ("block", "blocked"), ("schedule", "scheduled"), ("request_review", "review_requested"), ("request_changes", "changes_requested")])
@pytest.mark.parametrize("enabled", [True, False])
def test_settled_handoff_never_reenters_work(tmp_path, monkeypatch, tool, outcome, enabled):
    agent = _e2e_agent(tmp_path, monkeypatch)
    kb, conn, tid, rid = _board(tmp_path, monkeypatch, review=tool == "request_changes")
    hook = _hook(monkeypatch, enabled)
    from tests.agent.test_conversation_fallback_state import _tool_defs, _tool_call, _response
    name = "kanban_" + tool
    agent.tools = _tool_defs(name)
    agent.valid_tool_names = {name}
    calls = []
    def model(kwargs):
        calls.append(kwargs)
        if len(calls) > 2:
            pytest.fail("settled worker re-entered a tool-capable work iteration")
        if len(calls) == 2:
            return _e2e_response("handoff done")
        args = {"task_id": tid, "summary": "delivered"} if tool in {"complete", "request_review"} else {"task_id": tid, "reason": "needs input"}
        tc = _tool_call(name, "handoff")
        tc.function.arguments = json.dumps(args)
        return _response(content="", finish_reason="tool_calls", tool_calls=[tc])
    agent._interruptible_api_call = model
    try:
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            result = agent.run_conversation("finish assigned work")
        assert result["completed"] is True
        assert len(calls) == 2
        assert calls[0].get("tools"), "regression must exercise a tool-enabled loop"
        assert hook.call_count == 0
        run = kb.get_run(conn, rid)
        assert run.ended_at is not None and run.outcome == outcome
    finally:
        conn.close()


@pytest.mark.parametrize("control", ["failed", "stale_history", "wrong_run", "child"])
def test_unsettled_or_unowned_control_keeps_hook(tmp_path, monkeypatch, control):
    from agent.turn_stop_gates import apply_stop_gates
    from agent.delegation_context import non_dispatcher_owned_context
    from contextlib import nullcontext
    from tests.agent.test_pre_turn_finalize_hook import _StubAgent
    kb, conn, tid, rid = _board(tmp_path, monkeypatch)
    _hook(monkeypatch)
    monkeypatch.setattr("agent.turn_stop_gates._kanban_stop_nudge", lambda *a: None)
    messages = [{"role": "user", "content": "work"}, {"role": "tool", "name": "kanban_complete", "content": '{"error":"refused"}'}]
    try:
        if control == "stale_history":
            assert kb.request_review(conn, tid, summary="prior handoff", reviewer="reviewer", expected_run_id=rid)
            successor = kb.claim_review_task(conn, tid)
            assert successor is not None
            monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(successor.current_run_id))
            monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", successor.claim_lock)
            messages[-1]["content"] = '{"status":"review"}'
        if control == "failed":
            assert not kb.complete_task(conn, tid, summary="done", expected_run_id=rid + 999)
        if control in {"wrong_run", "child"}:
            assert kb.complete_task(conn, tid, summary="done", expected_run_id=rid)
        if control == "wrong_run":
            monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(rid + 999))
        with non_dispatcher_owned_context() if control == "child" else nullcontext():
            verdict = apply_stop_gates(_StubAgent(), {"role": "assistant", "content": "done"}, final_response="done", messages=messages, conversation_history=[], pending_verification_response=None, pending_verification_response_previewed=False)
        assert verdict.continue_turn is True
    finally:
        conn.close()


@pytest.mark.parametrize("with_db", [False, True])
def test_early_failure_strips_nudge_before_next_turn(tmp_path, monkeypatch, with_db):
    agent = _e2e_agent(tmp_path, monkeypatch)
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB"):
        monkeypatch.delenv(key, raising=False)
    if with_db:
        from hermes_state import SessionDB
        agent._session_db = SessionDB(db_path=tmp_path / "session.db")
    _hook(monkeypatch)
    answers = iter(["real candidate", "looping line\n" * 2000, "next answer", "next final"])
    requests = []
    def model(kwargs):
        requests.append(kwargs)
        if len(requests) == 2:
            assert any(m.get("content") == "internal plugin continuation" for m in kwargs["messages"])
        return _e2e_response(next(answers))
    agent._interruptible_api_call = model
    try:
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            first = agent.run_conversation("first request")
            assert first["completed"] is False and first["partial"] is True
            assert first["failure_reason"] == "truncated" and first["failure_retryable"] is True
            for history in (first["messages"], agent._session_messages):
                assert not any(m.get("_pre_turn_finalize_synthetic") for m in history)
                assert any(m.get("content") == "real candidate" for m in history)
            agent.run_conversation("next request", conversation_history=first["messages"])
        assert not any(m.get("content") == "internal plugin continuation" for m in requests[2]["messages"])
        if with_db:
            assert not any(m.get("content") == "internal plugin continuation" for m in agent._session_db.get_messages(agent.session_id))
    finally:
        if with_db:
            agent._session_db.close()
