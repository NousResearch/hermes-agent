"""A tool call whose arguments turn into an endless whitespace run aborts the stream as a transport
error (both wires retry it) instead of streaming until the output cap. Port of openclaw/openclaw#177659."""

import json
from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers_tool_calls import _ToolCallAccumulator
from agent.codex_runtime import _consume_codex_event_stream
from agent.repetition_guard import DegenerateToolArgumentsError


def _codex_events(arg_deltas):
    item = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "terminal", "arguments": ""}
    yield {"type": "response.output_item.added", "output_index": 0, "item": item}
    for delta in arg_deltas:
        yield {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": delta}
    yield {"type": "response.output_item.done", "output_index": 0,
           "item": {**item, "arguments": "".join(arg_deltas)}}
    yield {"type": "response.completed", "response": {"id": "r1", "status": "completed"}}


def _chat_delta(args):
    return SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(name="terminal", arguments=args))


RUNAWAY = ['{"command": "ls"'] + [" \n \n"] * 2000
# 300 KiB of real content: 5,000-space runs inside a JSON string are content, not a runaway.
BIG_OK = [json.dumps({"command": "echo " + ("x" * 60 + " " * 5000) * 60})[i:i + 512]
          for i in range(0, 310_000, 512)]


def test_whitespace_runaway_aborts_both_wires():
    with pytest.raises(DegenerateToolArgumentsError):
        _consume_codex_event_stream(_codex_events(RUNAWAY), model="m", abort_degenerate_args=True)
    acc = _ToolCallAccumulator(abort_degenerate_args=True)
    with pytest.raises(DegenerateToolArgumentsError):
        for delta in RUNAWAY:
            acc.feed(_chat_delta(delta))


def test_large_legitimate_arguments_and_replay_consumers_are_untouched():
    result = _consume_codex_event_stream(_codex_events(BIG_OK), model="m", abort_degenerate_args=True)
    assert result.output[0]["arguments"] == "".join(BIG_OK)
    acc = _ToolCallAccumulator(abort_degenerate_args=True)
    for delta in BIG_OK:
        acc.feed(_chat_delta(delta))
    assert acc.materialize()[0]["function"]["arguments"] == "".join(BIG_OK)
    # Relay's finalizer/accumulator replays frames without the flag and never raises.
    replay = _consume_codex_event_stream(_codex_events(RUNAWAY), model="m")
    assert replay.output[0]["arguments"].startswith('{"command": "ls"')
