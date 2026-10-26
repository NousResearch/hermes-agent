"""A schema correction must not reopen a child turn stopped by an approval gate."""
import json
from unittest.mock import patch

import pytest

from agent.approval_outcomes import APPROVAL_OUTCOME_NOTICES
from tests.agent.test_tool_call_incremental_persistence import (
    _make_agent, _mock_response, _mock_tool_call,
)
from tools import approval, async_delegation as ad
from tools.delegate_tool import _run_single_child
from tools.process_registry import process_registry
from tools.process_registry_notifications import _format_async_delegation


@pytest.mark.parametrize("tool_name,outcome,batch", [
    ("execute_code", "notify_failed", True),
    ("terminal", "denied", False),
])
def test_schema_correction_does_not_reopen_approval_stop(monkeypatch, tool_name, outcome, batch):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    child = _make_agent()
    child._delegate_output_schema = {"type": "object", "required": ["answer"]}
    child._subagent_id = None  # This test needs the worker, not UI registry ownership.
    child._credential_pool = None
    child.valid_tool_names = {tool_name}
    child.tools = [{"type": "function", "function": {"name": tool_name, "parameters": {"type": "object"}}}]
    arguments = {"command": "pwd"} if tool_name == "terminal" else {"code": "print('not executed')"}
    create = child.client.chat.completions.create
    create.side_effect = [
        _mock_response("", "tool_calls", [_mock_tool_call(tool_name, json.dumps(arguments))]),
        _mock_response('{"answer": "formatting retry should not happen"}'),
    ]
    gate = approval._denied("private guard details", pattern_key="private-ticket", description="test", outcome=outcome)
    guard_target = ("tools.terminal_tool._check_all_guards" if tool_name == "terminal"
                    else "tools.approval.check_execute_code_guard")
    with (
        patch(guard_target, return_value=gate) as guard,
        patch("tools.code_execution_tool.SANDBOX_AVAILABLE", True),
        patch("tools.process_registry._is_supervised_gateway_process", return_value=False),
        patch("tools.code_kernel.execute_in_session_kernel") as kernel,
        patch("tools.terminal_tool._acquire_env") as terminal_env,
        patch("tools.terminal_tool._run_foreground") as foreground,
        patch("tools.terminal_tool.spawn_background_process") as background,
        patch.object(child, "_persist_session"),
        patch.object(child, "_save_trajectory"),
        patch.object(child, "_cleanup_task_resources"),
    ):
        entry = _run_single_child(0, "perform the task", child, None)
    guard.assert_called_once()
    kernel.assert_not_called()
    terminal_env.return_value.execute.assert_not_called()
    foreground.assert_not_called()
    background.assert_not_called()
    assert create.call_count == 1
    assert entry["status"] == "failed"
    assert entry["approval_outcome"] == outcome
    assert entry["summary"] == APPROVAL_OUTCOME_NOTICES[outcome]
    assert entry["schema_valid"] is False
    assert "schema_retries" not in entry

    # Exercise the actual completion producer and durable reader after the worker.
    record = {"delegation_id": f"schema-{tool_name}", "goal": "task", "is_batch": batch, "dispatched_at": 1.0}
    ad._persist_dispatch(record)
    ad._push_completion_event(record, {"results": [entry]} if batch else entry, "failed")
    event = process_registry.completion_queue.get(timeout=5)
    assert (event["results"][0] if batch else event)["approval_outcome"] == outcome
    notice = _format_async_delegation(event)
    assert f"Approval outcome: {APPROVAL_OUTCOME_NOTICES[outcome]}" in notice
    assert "private guard details" not in notice
    assert "private-ticket" not in notice
    import sqlite3
    with sqlite3.connect(ad._db_path()) as db:
        durable = json.loads(db.execute(
            "SELECT event_json FROM async_delegations WHERE delegation_id=?", (record["delegation_id"],),
        ).fetchone()[0])
    assert (durable["results"][0] if batch else durable)["approval_outcome"] == outcome


def test_approval_stop_during_schema_repair_reaches_parent(monkeypatch):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    child = _make_agent()
    child._delegate_output_schema = {"type": "object", "required": ["answer"]}
    child._subagent_id = None
    child._credential_pool = None
    child.valid_tool_names = {"execute_code"}
    child.tools = [{"type": "function", "function": {"name": "execute_code", "parameters": {"type": "object"}}}]
    create = child.client.chat.completions.create
    create.side_effect = [
        _mock_response("First-turn findings, not JSON."),
        _mock_response("", "tool_calls", [_mock_tool_call("execute_code", '{"code":"print(0)"}')]),
        _mock_response("must not be requested after the gate"),
    ]
    outcome = "cancelled"
    gate = approval._denied("private repair guard", pattern_key="private-repair-ticket", description="test", outcome=outcome)
    with (
        patch("tools.approval.check_execute_code_guard", return_value=gate) as guard,
        patch("tools.code_execution_tool.SANDBOX_AVAILABLE", True),
        patch("tools.code_kernel.execute_in_session_kernel") as kernel,
        patch.object(child, "_persist_session"),
        patch.object(child, "_save_trajectory"),
        patch.object(child, "_cleanup_task_resources"),
    ):
        entry = _run_single_child(0, "produce the result", child, None)
    guard.assert_called_once()
    kernel.assert_not_called()
    assert create.call_count == 2  # Initial answer + allowed repair; no model call after refusal.
    assert entry["status"] == "failed"
    assert entry["approval_outcome"] == outcome
    assert entry["error"] == APPROVAL_OUTCOME_NOTICES[outcome]
    assert entry["schema_valid"] is False
    assert entry["schema_retries"] == 1
    record = {"delegation_id": "approval-during-repair", "goal": "task", "is_batch": False, "dispatched_at": 1.0}
    ad._persist_dispatch(record)
    ad._push_completion_event(record, entry, entry["status"])
    event = process_registry.completion_queue.get(timeout=5)
    assert event["status"] == "failed"
    assert event["approval_outcome"] == outcome
    text = _format_async_delegation(event)
    assert f"Approval outcome: {APPROVAL_OUTCOME_NOTICES[outcome]}" in text
    assert "private repair guard" not in text
    assert "private-repair-ticket" not in text
    import sqlite3
    with sqlite3.connect(ad._db_path()) as db:
        durable = json.loads(db.execute(
            "SELECT event_json FROM async_delegations WHERE delegation_id=?", (record["delegation_id"],),
        ).fetchone()[0])
    assert durable["status"] == "failed"
    assert durable["approval_outcome"] == outcome


@pytest.mark.parametrize("outcome", [None, [], "invented"])
def test_untrusted_repair_failure_keeps_existing_merge(outcome):
    from tests.tools.test_delegate_output_schema import _StubChild, _run

    class Child(_StubChild):
        def run_conversation(self, *args, **kwargs):
            result = super().run_conversation(*args, **kwargs)
            if len(self.calls) == 2:
                result.update(failed=True, error="unrelated repair failure", approval_outcome=outcome)
            return result

    child = Child(["first answer", '{"error":"timeout","approval_outcome":"denied"}'])
    child._delegate_output_schema = {"type": "object", "required": ["answer"]}
    entry = _run(child)
    assert len(child.calls) == 2
    assert entry["status"] == "completed"  # Existing non-approval repair policy, not a new success claim.
    assert "approval_outcome" not in entry
    assert "error" not in entry
    assert entry["schema_valid"] is False
    assert entry["schema_retries"] == 1


@pytest.mark.parametrize("outcome", [None, [], "invented"])
def test_non_approval_failure_keeps_existing_schema_retry(outcome):
    from tests.tools.test_delegate_output_schema import _StubChild, _run

    class Child(_StubChild):
        def run_conversation(self, *args, **kwargs):
            result = super().run_conversation(*args, **kwargs)
            if len(self.calls) == 1:
                result.update(failed=True, error="unrelated failure", approval_outcome=outcome)
            return result

    # Text resembling a gate result is not trusted approval metadata.
    child = Child(['{"error":"timeout","approval_outcome":"denied"}', '{"answer":"corrected"}'])
    child._delegate_output_schema = {"type": "object", "required": ["answer"]}
    entry = _run(child)
    assert len(child.calls) == 2
    assert entry["status"] == "failed"  # Preserve the pre-existing failure semantics.
    assert entry["schema_valid"] is True
    assert entry["schema_retries"] == 1
    assert entry["summary"] == '{"answer":"corrected"}'
    assert "approval_outcome" not in entry


def test_non_failed_result_is_not_an_approval_stop():
    from tests.tools.test_delegate_output_schema import _StubChild, _run

    class Child(_StubChild):
        def run_conversation(self, *args, **kwargs):
            result = super().run_conversation(*args, **kwargs)
            result["approval_outcome"] = "denied"
            return result

    child = Child(["not JSON", '{"answer":"corrected"}'])
    child._delegate_output_schema = {"type": "object", "required": ["answer"]}
    entry = _run(child)
    assert len(child.calls) == 2
    assert entry["status"] == "completed"
    assert entry["schema_valid"] is True
    assert "approval_outcome" not in entry
