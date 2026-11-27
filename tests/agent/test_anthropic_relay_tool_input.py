"""Regression tests for relay-rebuilt Anthropic tool_use inputs staying JSON (#133500)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("nemo_relay")

from agent.relay_llm import AnthropicStreamAccumulator
from agent.transports.anthropic import AnthropicTransport


def _tool_use_response(partial_json: str) -> object:
    accumulator = AnthropicStreamAccumulator()
    accumulator.observe({
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "call-1",
            "name": "read_file",
            "input": {},
        },
    })
    accumulator.observe({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    })
    accumulator.observe({"type": "message_delta", "delta": {"stop_reason": "tool_use"}})
    return accumulator.response()


def test_relay_tool_input_survives_normalization_verbatim():
    arguments = {
        "path": "report.txt",
        "nested": {"value": "kept"},
        "items": [None, False, 3],
        "_underscore_key": "preserved",
    }

    response = _tool_use_response(json.dumps(arguments))
    tool_calls = AnthropicTransport().normalize_response(response).tool_calls

    assert tool_calls[0].arguments == json.dumps(arguments)
    assert json.loads(tool_calls[0].arguments) == arguments


def test_relay_malformed_tool_input_keeps_parse_error_path():
    malformed = '{"path": "report.txt", "truncated":'

    response = _tool_use_response(malformed)
    tool_calls = AnthropicTransport().normalize_response(response).tool_calls

    # The unparsable partial stays the raw string; consumers hit their ordinary parse-error path.
    assert json.loads(tool_calls[0].arguments) == malformed
