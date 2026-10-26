"""Approval gate -> live turn -> child entry -> queued/durable completion."""
import json
import sqlite3
from unittest.mock import patch

import pytest

from tests.agent.test_tool_call_incremental_persistence import (
    _make_agent, _mock_response, _mock_tool_call,
)
from tools import approval, async_delegation as ad, delegate_tool_child_run as child_run
from tools.process_registry import process_registry
from tools.process_registry_notifications import _format_async_delegation


@pytest.mark.parametrize("outcome", ["denied", "timeout", "cancelled", "notify_failed", "blocked"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("tool_name", ["terminal", "execute_code"])
def test_live_turn_to_completion(tmp_path, monkeypatch, outcome, batch, tool_name):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    agent = _make_agent()
    agent.valid_tool_names = {tool_name}
    agent.tools = [{"type": "function", "function": {"name": tool_name, "parameters": {"type": "object"}}}]
    arguments = {"command": "pwd"} if tool_name == "terminal" else {"code": "print('must not execute')"}
    agent.client.chat.completions.create.side_effect = [
        _mock_response("", "tool_calls", [_mock_tool_call(tool_name, json.dumps(arguments))]),
        _mock_response("done"),
    ]
    gate = approval._denied("Do NOT retry", pattern_key="test", description="test", outcome=outcome)
    guard_target = ("tools.terminal_tool._check_all_guards" if tool_name == "terminal"
                    else "tools.approval.check_execute_code_guard")
    with (
        patch(guard_target, return_value=gate),
        patch("tools.code_execution_tool.SANDBOX_AVAILABLE", True),
        patch("tools.process_registry._is_supervised_gateway_process", return_value=False),
        patch("tools.code_kernel.execute_in_session_kernel") as kernel,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("run the command")
    kernel.assert_not_called()
    assert result.get("approval_outcome") == outcome
    assert result["failed"] is True
    assert agent.client.chat.completions.create.call_count == 1
    entry = child_run._build_result_entry(
        agent, result, task_index=0, duration=1.0,
        schema=child_run._SchemaOutcome(schema=None, valid=None, errors=[], retries=0),
    )
    assert entry["approval_outcome"] == outcome
    record = {"delegation_id": f"approval-{outcome}-{batch}", "goal": "task", "is_batch": batch, "dispatched_at": 1.0}
    ad._persist_dispatch(record)
    ad._push_completion_event(record, {"results": [entry]} if batch else entry, "failed")
    evt = process_registry.completion_queue.get(timeout=5)
    assert (evt["results"][0] if batch else evt)["approval_outcome"] == outcome
    text = _format_async_delegation(evt)
    from agent.approval_outcomes import APPROVAL_OUTCOME_NOTICES
    assert f"Approval outcome: {APPROVAL_OUTCOME_NOTICES[outcome]}" in text
    with sqlite3.connect(ad._db_path()) as db:
        row = db.execute("SELECT event_json FROM async_delegations WHERE delegation_id=?", (record["delegation_id"],)).fetchone()
    assert row is not None
    durable = json.loads(row[0])
    assert (durable["results"][0] if batch else durable)["approval_outcome"] == outcome


def test_error_builder_covers_all_gate_outcomes():
    from tools.code_execution_tool import _error_result
    for outcome in ("denied", "timeout", "cancelled", "notify_failed", "blocked"):
        assert json.loads(_error_result("blocked", approval_outcome=outcome))["approval_outcome"] == outcome
    assert "approval_outcome" not in json.loads(_error_result("blocked", approval_outcome="invented"))
