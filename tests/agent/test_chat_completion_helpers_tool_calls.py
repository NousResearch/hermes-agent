"""Slot assignment of streamed tool-call deltas whose id changes at a reused index."""

from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers_tool_calls import _ToolCallAccumulator
from agent.chat_completion_helpers_relay import RelayChatAccumulator


def _calls(*deltas, finish_reason="tool_calls"):
    acc = _ToolCallAccumulator()
    for tc_id, name, args in deltas:
        acc.feed(SimpleNamespace(index=0, id=tc_id, function=SimpleNamespace(name=name, arguments=args)))
    return [(e["function"]["name"], e["function"]["arguments"]) for _, e in sorted(
        acc.materialize(allow_replay=finish_reason in {"stop", "tool_calls", "function_call"}).items())]


def test_nameless_fragments_with_fresh_ids_stay_one_call():
    # LM Studio / Kimi: fresh id on every argument fragment, name only on the first (#5254).
    assert _calls(
        ("701601222", "terminal", ""),
        ("tool-call-1", None, '{"com'),
        ("tool-call-2", None, 'mand": '),
        ("tool-call-3", None, '"ls"}'),
    ) == [("terminal", '{"command": "ls"}')]


def test_same_name_fresh_id_splits_only_after_complete_arguments():
    # A name resent on each fragment continues the call (MiniMax); a same-name delta after
    # closed arguments is a second parallel call (Ollama), even when the first is only repairable
    # JSON (trailing comma) or has brackets inside a string; a different name always splits.
    assert _calls(
        ("mm-1", "terminal", '{"command": '),
        ("mm-2", "terminal", '"echo \\"}{\\""}'),
        ("mm-3", "terminal", '{"command": "pwd",}'),
        ("mm-4", "terminal", '{"command": "ls"}'),
        ("mm-5", "read_file", "{}"),
    ) == [
        ("terminal", '{"command": "echo \\"}{\\""}'),
        ("terminal", '{"command": "pwd",}'),
        ("terminal", '{"command": "ls"}'),
        ("read_file", "{}"),
    ]


def test_complete_argument_replay_replaces_streamed_prefix():
    # Some OpenAI-compatible gateways send an entire valid argument object after
    # partial deltas. Joining all chunks makes even a valid tool call unparseable.
    assert _calls(
        ("call-1", "manage_connections", ""),
        (None, None, '{"action": "'),
        (None, None, "display"),
        (None, None, ', "connectors": [""]'),
        (None, None, "WhatsApp"),
        (None, None, '{"action": "display", "connectors": ["WhatsApp"]}'),
    ) == [("manage_connections", '{"action": "display", "connectors": ["WhatsApp"]}')]


def test_complete_nested_object_fragment_does_not_replace_truncated_outer_call():
    assert _calls(
        ("call-1", "terminal", '{"outer": '),
        (None, None, '{"outer": 1}'),
    ) == [("terminal", '{"outer": {"outer": 1}')]


def test_complete_argument_replay_without_finish_reason_stays_incomplete():
    assert _calls(
        ("call-1", "manage_connections", ""),
        (None, None, '{"action": "'),
        (None, None, "display"),
        (None, None, ', "connectors": [""]'),
        (None, None, "WhatsApp"),
        (None, None, '{"action": "display", "connectors": ["WhatsApp"]}'),
        finish_reason=None,
    ) == [("manage_connections", '{"action": "display, "connectors": [""]WhatsApp'
          '{"action": "display", "connectors": ["WhatsApp"]}')]


def test_nested_object_after_array_comma_is_not_a_replay():
    assert _calls(
        ("call-1", "terminal", '{"items": [1, '),
        (None, None, '{"items": [1, 2]}'),
    ) == [("terminal", '{"items": [1, {"items": [1, 2]}')]


@pytest.mark.parametrize("terminal", ["TOOL_CALLS", "FUNCTION_CALL", "end"])
def test_relay_replay_requires_finish_reason(terminal):
    relay = RelayChatAccumulator()
    for part in ('{"action": "', 'display', ', "connectors": [""]', 'WhatsApp',
                 '{"action": "display", "connectors": ["WhatsApp"]}'):
        relay.observe({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1",
            "function": {"name": "manage_connections", "arguments": part}}]}, "finish_reason": None}]})
    raw = relay.finalize()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert raw != '{"action": "display", "connectors": ["WhatsApp"]}'
    relay.observe({"choices": [{"delta": {}, "finish_reason": terminal}]})
    finalized = relay.finalize()["choices"][0]
    repaired = finalized["message"]["tool_calls"][0]["function"]["arguments"]
    assert repaired == '{"action": "display", "connectors": ["WhatsApp"]}'
    assert finalized["finish_reason"] == ("stop" if terminal == "end" else "tool_calls")
