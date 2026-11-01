"""Selected channel tools are checked at the common pre-dispatch seam."""

import json
from types import SimpleNamespace

from agent import tool_executor


def _call(monkeypatch, allowed, name):
    emitted = []
    executed = []
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", lambda *a, **kw: emitted.append(kw))
    agent = SimpleNamespace(
        _protected_channel_tool_names=allowed,
        _tool_guardrails=SimpleNamespace(before_call=lambda *_: SimpleNamespace(allows_execution=False)),
        _guardrail_block_result=lambda _: json.dumps({"error": "ordinary guardrail"}),
    )
    state = tool_executor._ManagedToolResult(None, {}, [], False, False)
    ref = tool_executor._ToolCallRef(name, {}, "task", "call", [])
    result = tool_executor._dispatch_authorized_once(
        agent, state, ref, execute=lambda _: executed.append(name), scope_block=None,
        display_index=None, begin_execution=None, authorization_gate=None)
    return json.loads(result), emitted, executed


def test_protected_channel_blocks_a_direct_unadvertised_tool_before_hooks_or_dispatch(monkeypatch):
    result, emitted, executed = _call(monkeypatch, frozenset({"approved_plugin_tool"}), "terminal")
    assert "unavailable in this channel" in result["error"]
    assert emitted[0]["error_type"] == "tool_scope_block"
    assert executed == []


def test_selected_tool_and_unprotected_channel_continue_to_ordinary_policy(monkeypatch):
    for grant, name in ((frozenset({"approved_plugin_tool"}), "approved_plugin_tool"), (None, "terminal")):
        result, emitted, executed = _call(monkeypatch, grant, name)
        assert result == {"error": "ordinary guardrail"}
        assert emitted[0]["error_type"] == "guardrail_block"
        assert executed == []
