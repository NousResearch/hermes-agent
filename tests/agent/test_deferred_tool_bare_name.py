"""A deferred tool called by its real name runs through the tool_call bridge (#130153)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from run_agent import AIAgent

_ARGS = {"action": "poll", "session_id": "proc_x"}


def _validate(arguments, *, valid_tool_names=("terminal", "tool_search", "tool_describe", "tool_call"),
              enabled_toolsets=None, provider_data=None):
    """Real validation on an agent whose tool search deferred process_manage behind the bridge."""
    from agent.turn_tool_validation import validate_tool_calls

    agent = AIAgent.__new__(AIAgent)
    agent.provider, agent.model, agent.session_id, agent.tools = "nous", "m", "s1", []
    agent.valid_tool_names = set(valid_tool_names)
    agent.log_prefix, agent._base_url, agent.api_mode = "", "https://api.example.invalid/v1", "chat_completions"
    agent._invalid_tool_retries = agent._invalid_json_retries = 0
    agent.enabled_toolsets, agent.disabled_toolsets = enabled_toolsets, None
    agent._vprint = lambda *a, **k: None
    tc = SimpleNamespace(id="call_1", type="function",
                         function=SimpleNamespace(name="process_manage", arguments=arguments))
    messages = []
    message = SimpleNamespace(content="", tool_calls=[tc], provider_data=provider_data)
    verdict = validate_tool_calls(agent, message, "tool_calls",
                                  messages=messages, conversation_history=[], api_call_count=1,
                                  effective_task_id="t1")
    return verdict, tc, [m["content"] for m in messages if m.get("role") == "tool"], agent


@pytest.mark.parametrize("arguments, expected, executor_name", [
    (json.dumps(_ARGS), _ARGS, "process_manage"),
    ("", {}, "tool_call"),  # the bridge's schema probe refuses the missing 'action'
])
def test_real_name_is_rewritten_into_the_bridge_call(arguments, expected, executor_name):
    from agent.tool_executor import _parse_tool_call

    verdict, tc, errors, agent = _validate(arguments)

    assert verdict.action == "ok"
    assert tc.function.name == "tool_call"
    assert json.loads(tc.function.arguments) == {"calls": [{"name": "process_manage", "arguments": expected}]}
    assert errors == []
    parsed = _parse_tool_call(agent, tc)
    assert parsed.name == executor_name
    assert (parsed.scope_block is None) == (executor_name == "process_manage")


def test_native_replay_names_the_bridge_call():
    """Signed-thinking turns replay from the ordered sidecars; the routed call must replay as tool_call."""
    from agent.anthropic_message_convert import _convert_assistant_message
    from agent.bedrock_adapter import _assistant_blocks

    provider_data = {
        "anthropic_content_blocks": [{"type": "thinking", "thinking": "poll it", "signature": "sig"},
                                     {"type": "tool_use", "id": "call_1", "name": "process_manage", "input": _ARGS}],
        "bedrock_content_blocks": [{"reasoningContent": {"text": "poll it", "signature": "sig"}},
                                   {"toolUse": {"toolUseId": "call_1", "name": "process_manage", "input": _ARGS}}],
    }
    _, tc, _, _ = _validate(json.dumps(_ARGS), provider_data=provider_data)
    stored = {"role": "assistant", "content": "", **provider_data, "tool_calls": [
        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]}
    routed = {"calls": [{"name": "process_manage", "arguments": _ARGS}]}

    assert [b for b in _convert_assistant_message(stored)["content"] if b["type"] == "tool_use"] == [
        {"type": "tool_use", "id": "call_1", "name": "tool_call", "input": routed}]
    assert [b["toolUse"] for b in _assistant_blocks(stored, "") if "toolUse" in b] == [
        {"toolUseId": "call_1", "name": "tool_call", "input": routed}]


@pytest.mark.parametrize("arguments, kwargs", [
    (json.dumps(_ARGS), {"enabled_toolsets": ["file"]}),
    (json.dumps(_ARGS), {"valid_tool_names": ("terminal",)}),
    ('{"action": ', {}),
], ids=["out_of_scope", "bridge_off", "unparseable_args"])
def test_unroutable_call_is_still_rejected(arguments, kwargs):
    verdict, tc, errors, _ = _validate(arguments, **kwargs)

    assert verdict.action == "continue"
    assert tc.function.name == "process_manage"
    assert errors and errors[0].startswith("Tool 'process_manage' does not exist")
