"""Regression coverage for #122559 canonical compression state."""

from __future__ import annotations

import json
from unittest.mock import patch

from agent.compression_marker import _COMPRESSION_MARKER_PREFIX
from agent.context_compressor import ContextCompressor


def _call(call_id: str, payload: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps({"path": "/tmp/x.py", "content": payload}),
            },
        }],
    }


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "head request"},
        _call("head_call", "A" * 4_000),
        {"role": "tool", "tool_call_id": "head_call", "content": "R" * 10_000},
        {"role": "user", "content": "middle 1"},
        {"role": "assistant", "content": "middle answer 1"},
        {"role": "user", "content": "middle 2"},
        {"role": "assistant", "content": "middle answer 2"},
        {"role": "user", "content": "tail request"},
        {"role": "assistant", "content": "tail answer"},
    ]


def _compressor() -> ContextCompressor:
    return ContextCompressor(
        model="test/model",
        config_context_length=200_000,
        protect_first_n=3,
        protect_last_n=2,
        quiet_mode=True,
        tail_mode="legacy",
    )


def test_structural_noop_returns_canonical_tool_arguments() -> None:
    compressor = _compressor()
    messages = _messages()
    original_args = messages[1]["tool_calls"][0]["function"]["arguments"]

    with patch.object(compressor, "_compress_window", return_value=(3, 3)):
        result = compressor.compress(messages, current_tokens=100_000)

    assert result[1]["tool_calls"][0]["function"]["arguments"] == original_args
    assert _COMPRESSION_MARKER_PREFIX not in json.dumps(result, ensure_ascii=False)


def test_successful_compaction_carries_canonical_head_not_pruned_working_copy() -> None:
    compressor = _compressor()
    messages = _messages()
    original_args = messages[1]["tool_calls"][0]["function"]["arguments"]
    original_result = messages[2]["content"]

    with (
        patch.object(compressor, "_compress_window", return_value=(3, 7)),
        patch.object(compressor, "_generate_summary", return_value="Summary of middle turns."),
    ):
        result = compressor.compress(messages, current_tokens=100_000, force=True)

    head_call = next(
        message for message in result
        if message.get("role") == "assistant"
        and any(call.get("id") == "head_call" for call in message.get("tool_calls", []))
    )
    head_tool = next(
        message for message in result
        if message.get("role") == "tool" and message.get("tool_call_id") == "head_call"
    )
    assert head_call["tool_calls"][0]["function"]["arguments"] == original_args
    assert head_tool["content"] == original_result
    assert _COMPRESSION_MARKER_PREFIX not in head_call["tool_calls"][0]["function"]["arguments"]
