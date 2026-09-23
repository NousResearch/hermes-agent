"""Tests for _ToolCallAccumulator parallel streaming and extra_content propagation."""

from types import SimpleNamespace
from agent.chat_completion_helpers import _ToolCallAccumulator


def test_tool_call_accumulator_propagates_thought_signature_to_parallel_siblings():
    """Gemini emits thought_signature on one delta; sibling parallel tool calls must inherit it."""
    acc = _ToolCallAccumulator()

    # Delta 0: tool call 0 starts with function name
    acc.feed(SimpleNamespace(index=0, id="call_0", function=SimpleNamespace(name="read_file", arguments='{"path": ')))

    # Delta 1: tool call 1 starts with function name
    acc.feed(SimpleNamespace(index=1, id="call_1", function=SimpleNamespace(name="web_search", arguments='{"query": ')))

    # Delta 2: tool call 0 receives extra_content (thought signature) and args
    acc.feed(SimpleNamespace(
        index=0,
        function=SimpleNamespace(name=None, arguments='"foo.py"}'),
        extra_content={"google": {"thought_signature": "gemini_sig_123"}},
    ))

    # Delta 3: tool call 1 finishes args without receiving extra_content delta
    acc.feed(SimpleNamespace(index=1, function=SimpleNamespace(name=None, arguments='"hermes"}')))

    res = acc.materialize()
    assert len(res) == 2
    assert res[0]["function"]["name"] == "read_file"
    assert res[0]["function"]["arguments"] == '{"path": "foo.py"}'
    assert res[0]["extra_content"] == {"google": {"thought_signature": "gemini_sig_123"}}

    # Sibling tool call 1 must inherit the shared thought signature
    assert res[1]["function"]["name"] == "web_search"
    assert res[1]["function"]["arguments"] == '{"query": "hermes"}'
    assert res[1]["extra_content"] == {"google": {"thought_signature": "gemini_sig_123"}}
