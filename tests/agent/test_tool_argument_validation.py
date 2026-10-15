"""Inline dispatch preserves supported aliases but rejects silently ignored keys."""

import json
from types import SimpleNamespace

import pytest

import model_tools  # Discover real schemas, including agent-level tools.
from agent import tool_executor as te


@pytest.mark.parametrize("args,invalid", [
    ({"goal": "Summarize", "context": "text"}, False),
    ({"goal": "Summarize", "invented_parameter": True}, True),
])
def test_inline_arguments_checked_before_execution_without_stalling_start_order(monkeypatch, args, invalid):
    calls, starts = [], []
    agent = SimpleNamespace(_tool_guardrails=SimpleNamespace(
        before_call=lambda *a: SimpleNamespace(allows_execution=True)), _touch_activity=lambda *a: None)
    monkeypatch.setattr(te, "_pre_tool_block", lambda agent, ref: (None, ref.args))
    monkeypatch.setattr(te, "_begin_tool_execution", lambda *a: None)
    ref = te._ToolCallRef("delegate_task", args, "task", "call", [])
    state = te._ManagedToolResult(None, args, [], False, False)

    def begin(callback=None):
        starts.append(callback)
        if callback:
            callback()

    result = te._dispatch_authorized_once(
        agent, state, ref, execute=lambda actual: calls.append(actual) or '{"ok": true}',
        scope_block=None, display_index=None, begin_execution=begin, authorization_gate=None)
    assert len(starts) == 1
    if invalid:
        assert "NOT invoked" in json.loads(result)["error"]
        assert calls == []
        assert starts == [None]
    else:
        assert json.loads(result) == {"ok": True}
        assert calls == [args]


def test_inline_validation_checks_arguments_after_hook_rewrites(monkeypatch):
    calls = []
    agent = SimpleNamespace(_tool_guardrails=SimpleNamespace(
        before_call=lambda *a: SimpleNamespace(allows_execution=True)))
    args = {"goal": "Summarize"}
    monkeypatch.setattr(te, "_pre_tool_block", lambda agent, ref: (
        None, {**ref.args, "invented_parameter": True}))
    ref = te._ToolCallRef("delegate_task", args, "task", "call", [])
    state = te._ManagedToolResult(None, args, [], False, False)
    result = te._dispatch_authorized_once(
        agent, state, ref, execute=lambda actual: calls.append(actual), scope_block=None,
        display_index=None, begin_execution=None, authorization_gate=None)
    assert "invented_parameter" in json.loads(result)["error"]
    assert state.args == {**args, "invented_parameter": True}
    assert calls == []
