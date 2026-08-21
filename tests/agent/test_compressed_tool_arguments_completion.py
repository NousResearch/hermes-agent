import json
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.tool_argument_integrity import INCOMPLETE_TOOL_ARGUMENTS_KEY


def compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(model="test", quiet_mode=True)


def call(call_id, payload):
    return {"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": json.dumps(payload)}}


def test_pruning_only_marks_completed_large_tool_calls():
    complete = call("done", {"command": "x" * 1000})
    pending = call("pending", {"command": "y" * 1000})
    original_pending = pending["function"]["arguments"]
    messages = [
        {"role": "assistant", "tool_calls": [complete]},
        {"role": "tool", "tool_call_id": "done", "content": "ok"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "tool_calls": [pending]},
        {"role": "user", "content": "interrupted"},
    ]
    pruned, _ = compressor()._prune_old_tool_results(messages, protect_tail_count=0)
    completed_args = json.loads(pruned[0]["tool_calls"][0]["function"]["arguments"])
    assert INCOMPLETE_TOOL_ARGUMENTS_KEY in completed_args
    assert pruned[3]["tool_calls"][0]["function"]["arguments"] == original_pending
