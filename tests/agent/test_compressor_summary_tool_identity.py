"""Summarizer input keeps tool identity: calls carry ``#N`` and results repeat it with the tool name.

Before, a turn firing several calls of the same tool produced back-to-back
``[TOOL RESULT <opaque id>]`` blobs the summarizer could only pair with the
``name(args)`` lines by position, and a failed ``read_file`` was
indistinguishable from a failed ``terminal`` once the id was the only label.
Ported from PrimeIntellect-ai/prime-agent#2424.
"""

from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


@pytest.fixture()
def compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(model="test/model", threshold_percent=0.85, protect_first_n=2,
                                 protect_last_n=2, quiet_mode=True)


def _call(call_id: str, name: str, args: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def test_same_tool_called_twice_pairs_results_by_index(compressor):
    turns = [
        {"role": "assistant", "content": "", "tool_calls": [
            _call("call_" + "a" * 60, "read_file", '{"path": "a.py"}'),
            _call("call_" + "b" * 60, "read_file", '{"path": "b.py"}'),
        ]},
        {"role": "tool", "tool_call_id": "call_" + "a" * 60, "name": "read_file", "content": "contents of a"},
        {"role": "tool", "tool_call_id": "call_" + "b" * 60, "name": "read_file", "content": '{"error": "b.py missing"}'},
    ]
    out = compressor._serialize_for_summary(turns)
    assert '  #1 read_file({"path": "a.py"})' in out
    assert '  #2 read_file({"path": "b.py"})' in out
    assert "[TOOL RESULT #1 read_file]: contents of a" in out
    assert '[TOOL RESULT #2 read_file]: {"error": "b.py missing"}' in out
    # The opaque provider id never reaches the summarizer once the call is indexed.
    assert "a" * 60 not in out


def test_result_without_a_serialized_call_keeps_id_and_name(compressor):
    turns = [{"role": "tool", "tool_call_id": "orphan-1", "name": "terminal", "content": "ok"}]
    assert "[TOOL RESULT orphan-1 terminal]: ok" in compressor._serialize_for_summary(turns)
