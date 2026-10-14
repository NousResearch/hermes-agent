"""Tool result chaining (#78061): a tool can consume a prior tool's output via
``{{tool_result:<call_id>.<field>}}`` without the model re-emitting it."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.tool_result_chaining import (
    fold_tool_result_hint,
    remember_tool_result,
    resolve_tool_result_refs,
    tool_result_reference_hint,
)

_HINT = "\n\n[Reusable tool result: {{tool_result:t1.ok}}]"


def test_resolves_references_into_tool_args():
    agent = SimpleNamespace()
    remember_tool_result(agent, "call_1", '{"data":"abc","nested":{"x":7},"items":["a","b"]}')
    assert resolve_tool_result_refs(agent, {"image_url": "{{tool_result:call_1.data}}"}) == {"image_url": "abc"}
    assert resolve_tool_result_refs(agent, {"path": "{{tool_result:call_1.nested.x}}"}) == {"path": "7"}
    assert resolve_tool_result_refs(agent, {"index": "{{tool_result:call_1.items.1}}"}) == {"index": "b"}
    # Nested containers resolve to compact JSON strings (args stay strings).
    assert resolve_tool_result_refs(agent, {"cfg": "{{tool_result:call_1.nested}}"}) == {"cfg": '{"x": 7}'}


def test_hint_advertises_usable_reference_path():
    hint = tool_result_reference_hint("call_1", '{"ok":true}')
    assert "{{tool_result:call_1.ok}}" in hint
    # MCP-style wrapper {"result": {...}} reveals the nested path.
    hint = tool_result_reference_hint("call_1", '{"result": {"data": "x"}}')
    assert "{{tool_result:call_1.result.data}}" in hint
    # A top-level list advertises its first index (a resolvable path).
    hint = tool_result_reference_hint("call_1", '[{"a": 1}]')
    assert "{{tool_result:call_1.0}}" in hint
    # Non-structured results get no hint.
    assert tool_result_reference_hint("call_1", "plain text") == ""


def test_unresolvable_and_plain_values_pass_through():
    agent = SimpleNamespace()
    remember_tool_result(agent, "call_1", '{"data":"abc"}')
    # Unknown call id leaves the reference untouched (no crash).
    assert resolve_tool_result_refs(agent, {"x": "{{tool_result:missing.a}}"}) == {"x": "{{tool_result:missing.a}}"}
    # Plain strings pass through.
    assert resolve_tool_result_refs(agent, {"x": "hello"}) == {"x": "hello"}


def test_ref_map_bounds_by_entries_and_total_bytes(monkeypatch):
    import agent.tool_result_chaining as trc
    agent = SimpleNamespace()
    for i in range(40):
        remember_tool_result(agent, f"c{i}", json.dumps({"i": i}))
    refs = agent._tool_result_refs
    assert len(refs) == trc._MAX_TOOL_RESULT_REFS
    assert "c39" in refs and "c0" not in refs  # oldest evicted first

    agent2 = SimpleNamespace()
    monkeypatch.setattr(trc, "_MAX_TOOL_RESULT_REF_TOTAL_BYTES", 120)
    remember_tool_result(agent2, "a", json.dumps({"x": "a" * 60}))
    remember_tool_result(agent2, "b", json.dumps({"x": "b" * 60}))
    assert "b" in agent2._tool_result_refs and "a" not in agent2._tool_result_refs


def test_fold_helper_shapes():
    assert fold_tool_result_hint("body", _HINT) == "body" + _HINT
    assert fold_tool_result_hint("body", None) == "body"
    assert fold_tool_result_hint([{"type": "text", "text": "x"}], _HINT) == [
        {"type": "text", "text": "x"}, {"type": "text", "text": _HINT.strip()}]
    assert fold_tool_result_hint([], _HINT, part_type="input_text") == [{"type": "input_text", "text": _HINT.strip()}]
    # Other content shapes pass through unchanged.
    assert fold_tool_result_hint({"raw": 1}, _HINT) == {"raw": 1}
    assert fold_tool_result_hint(None, _HINT) is None


def test_mcp_double_encoded_result_resolves():
    """MCP tool results arrive double-encoded: the wrapper stores
    {"result": "<json-string>", "structuredContent": {"result": "<json-string>"}}
    (tools/mcp_tool_handlers.py::_render_call_tool_result). The walk must re-parse
    string leaves so chained paths like .result.data resolve against the real payload.
    """
    agent = SimpleNamespace()
    remember_tool_result(agent, "call_mcp", json.dumps({
        "result": '{"folders": ["a", "b"], "count": 2}',
        "structuredContent": {"result": '{"folders": ["a", "b"], "count": 2}'},
    }))

    # .result.folders walks into the string leaf, re-parses, and resolves.
    assert resolve_tool_result_refs(
        agent, {"image_url": "{{tool_result:call_mcp.result.folders}}"}
    ) == {"image_url": '["a", "b"]'}
    assert resolve_tool_result_refs(
        agent, {"image_url": "{{tool_result:call_mcp.result.count}}"}
    ) == {"image_url": "2"}
    # structuredContent branch resolves the same way.
    assert resolve_tool_result_refs(
        agent, {"image_url": "{{tool_result:call_mcp.structuredContent.result.folders}}"}
    ) == {"image_url": '["a", "b"]'}
    # Unresolvable refs pass through untouched (contract preserved).
    assert resolve_tool_result_refs(
        agent, {"x": "{{tool_result:call_mcp.result.missing}}"}
    ) == {"x": "{{tool_result:call_mcp.result.missing}}"}
    # Non-JSON string leaf: pass through untouched, no crash.
    remember_tool_result(agent, "call_plain", json.dumps({"result": "not json"}))
    assert resolve_tool_result_refs(
        agent, {"x": "{{tool_result:call_plain.result.nope}}"}
    ) == {"x": "{{tool_result:call_plain.result.nope}}"}
    # The hint prefers structuredContent (the machine-oriented branch) and
    # reveals a working path for the double-encoded shape.
    hint = tool_result_reference_hint("call_mcp", json.dumps({
        "result": '{"folders": ["a", "b"], "count": 2}',
        "structuredContent": {"result": '{"folders": ["a", "b"], "count": 2}'},
    }))
    assert "{{tool_result:call_mcp.structuredContent.result.field}}" in hint


def test_wire_folds_hint_into_model_visible_text():
    """Every wire converter folds the sibling key into model-visible text and never
    leaks the key itself."""
    tool_msg = {"role": "tool", "tool_call_id": "t1", "content": '{"ok": true}', "_tool_result_hint": _HINT}
    pair = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        tool_msg,
    ]
    # chat-completions transport
    from agent.transports.chat_completions import ChatCompletionsTransport
    wire = ChatCompletionsTransport().convert_messages([dict(tool_msg)], model="test-model")
    assert "{{tool_result:t1.ok}}" in wire[0]["content"] and "_tool_result_hint" not in wire[0]
    # anthropic
    from agent.anthropic_message_convert import convert_messages_to_anthropic
    _system, msgs = convert_messages_to_anthropic([dict(m) for m in pair])
    block = next(b for m in msgs for b in m["content"] if b.get("type") == "tool_result")
    assert "{{tool_result:t1.ok}}" in block["content"]
    # bedrock
    from agent.bedrock_adapter import convert_messages_to_converse
    _system, converse = convert_messages_to_converse([dict(tool_msg)])
    text = converse[0]["content"][0]["toolResult"]["content"][0]["text"]
    assert "{{tool_result:t1.ok}}" in text
    # codex responses
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    items = _chat_messages_to_responses_input([dict(tool_msg)])
    out = next(i for i in items if i.get("type") == "function_call_output")
    assert "{{tool_result:t1.ok}}" in out["output"]
    # gemini native: the structured payload keeps its shape with a namespaced hint
    # key; the output fallback appends the hint to the text.
    from agent.gemini_native_adapter import _translate_tool_result_to_gemini
    fr = _translate_tool_result_to_gemini(dict(tool_msg))["functionResponse"]
    assert fr["response"]["ok"] is True
    assert "{{tool_result:t1.ok}}" in fr["response"]["_hermes_tool_result_hint"]
    fr_plain = _translate_tool_result_to_gemini(
        {"role": "tool", "tool_call_id": "t1", "content": "plain", "_tool_result_hint": _HINT}
    )["functionResponse"]
    assert "{{tool_result:t1.ok}}" in fr_plain["response"]["output"]


def _tc(name="echo_tool", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.fixture()
def agent():
    a = AIAgent(
        api_key="test-key-1234567890",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        api_mode="chat_completions",
        model="test-model",
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
    )
    a.client = MagicMock()
    return a


def test_sequential_path_resolves_refs_before_dispatch(agent):
    """The default single-call path (sequential) must resolve refs.

    Regression guard for the review finding that resolution only fired on
    the concurrent path while the sequential executor passed
    {{tool_result:...}} through verbatim. A single tool call routes to
    execute_tool_calls_sequential, so the dispatched args must be resolved.
    """
    seen = {}

    def fake_handle(name, args, task_id, **kwargs):
        seen["args"] = args
        return json.dumps({"ok": True})

    remember_tool_result(agent, "call_abc123", '{"data": "iVBORw0KGgoAAA"}')

    msg = SimpleNamespace(
        content="",
        tool_calls=[_tc(arguments='{"image_url": "{{tool_result:call_abc123.data}}"}')],
    )
    messages = []

    with patch("model_tools.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls(msg, messages, "task-1")

    assert seen["args"] == {"image_url": "iVBORw0KGgoAAA"}, seen
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert tool_msgs
    # Content stays a parseable JSON string (JSON contract); the ref hint
    # rides as a sibling key, injected for the model only at the wire.
    tool_msg = tool_msgs[-1]
    assert json.loads(tool_msg["content"]) == {"ok": True}
    assert "{{tool_result:" in tool_msg.get("_tool_result_hint", "")
    # The wire-time injection (ChatCompletionsTransport.convert_messages)
    # folds the sibling key into the model-visible content.
    from agent.transports.chat_completions import ChatCompletionsTransport
    wire = ChatCompletionsTransport().convert_messages([tool_msg], model="test-model")
    assert "{{tool_result:" in wire[0]["content"]
    assert tool_msg["tool_call_id"] in wire[0]["content"]


def test_commit_gating_blocked_and_multimodal_get_no_hint(agent):
    """A blocked call never ran and a multimodal result's content becomes image
    blocks - neither may advertise a reusable reference."""
    from agent.tool_executor import _ToolCallRef, _budget_for_agent, _commit_tool_result

    def commit(result, *, blocked=False):
        messages = []
        ref = _ToolCallRef("echo_tool", {"x": 1}, "task-1", "t1", [])
        committed = _commit_tool_result(
            agent, messages, ref, result,
            budget=_budget_for_agent(agent), tool_duration=0.1, is_error=False, blocked=blocked,
            effect_disposition="none" if blocked else None,
        )
        assert committed is not None
        return messages[-1]

    assert "_tool_result_hint" in commit('{"ok": true}')
    assert "_tool_result_hint" not in commit('{"error": "blocked by policy"}', blocked=True)
    assert "_tool_result_hint" not in commit(
        {"_multimodal": True, "content": [{"type": "text", "text": "see image"}], "text_summary": "see image"}
    )


def test_summary_path_keeps_hint_for_the_wire(agent):
    """The max-iterations summary build must not sweep the hint away before the
    transport folds it (the underscore sweep in _iteration_summary_api_messages
    exempts the ref hint; the transport folds it at kwargs build)."""
    from agent.chat_completion_helpers import _build_api_kwargs_for_mode, _iteration_summary_api_messages
    agent._cached_system_prompt = "SYS"
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": '{"ok": true}', "_tool_result_hint": _HINT},
    ]
    msgs = _iteration_summary_api_messages(agent, [dict(m) for m in history])
    tool_msg = next(m for m in msgs if m.get("role") == "tool")
    assert "_tool_result_hint" in tool_msg  # survives the underscore sweep
    assert "{{tool_result:t1.ok}}" not in tool_msg["content"]  # content stays parseable JSON
    kwargs = _build_api_kwargs_for_mode(agent, msgs)
    wire_tool = next(m for m in kwargs["messages"] if m.get("role") == "tool")
    assert "_tool_result_hint" not in wire_tool
    assert "{{tool_result:t1.ok}}" in wire_tool["content"]
