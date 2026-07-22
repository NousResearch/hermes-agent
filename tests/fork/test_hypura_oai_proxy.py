"""Behavior tests for the Hypura OpenAI compatibility proxy."""

from __future__ import annotations

import json

import pytest

from fork.extensions.hypura_oai_proxy import (
    _build_hypura_generate_payload,
    _openai_generate_response,
)


def test_generate_payload_renders_messages_and_tools() -> None:
    body = {
        "model": "Agents A1 4B",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Add two and three."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        "required": ["a", "b"],
                    },
                },
            }
        ],
        "temperature": 0,
        "max_tokens": 128,
        "stream": True,
    }

    payload = _build_hypura_generate_payload(body, "fallback")

    assert payload["model"] == "Agents A1 4B"
    assert payload["stream"] is False  # proxy buffers to parse tool calls safely
    assert payload["options"]["temperature"] == 0
    assert payload["options"]["num_predict"] == 128
    assert "<|im_start|>system" in payload["prompt"]
    assert "Be concise." in payload["prompt"]
    assert '"name":"add"' in payload["prompt"]
    assert "<|im_start|>assistant" in payload["prompt"]


def test_generate_payload_rejects_missing_messages() -> None:
    with pytest.raises(Exception) as exc:
        _build_hypura_generate_payload({"model": "m"}, "fallback")
    assert getattr(exc.value, "status_code", None) == 400


def test_openai_response_extracts_tool_call_after_reasoning() -> None:
    hypura = {
        "response": (
            "<think>Need to use the tool.</think>\n"
            '{"tool_calls":[{"name":"add","arguments":{"a":2,"b":3}}]}'
        ),
        "prompt_eval_count": 10,
        "eval_count": 7,
    }

    result = _openai_generate_response(hypura, "chatcmpl-test", 123, "Agents A1 4B")

    message = result["choices"][0]["message"]
    assert message["content"] == ""
    assert message["reasoning_content"] == "Need to use the tool."
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    call = message["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "add"
    assert json.loads(call["function"]["arguments"]) == {"a": 2, "b": 3}
    assert result["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 7,
        "total_tokens": 17,
    }


def test_openai_response_keeps_plain_text_and_strips_think_block() -> None:
    hypura = {"response": "<think>internal</think>\nHello from Hypura"}

    result = _openai_generate_response(hypura, "chatcmpl-test", 123, "model")

    message = result["choices"][0]["message"]
    assert message["content"] == "Hello from Hypura"
    assert message["reasoning_content"] == "internal"
    assert "tool_calls" not in message
    assert result["choices"][0]["finish_reason"] == "stop"
