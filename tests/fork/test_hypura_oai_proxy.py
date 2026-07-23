"""Behavior tests for the Hypura OpenAI compatibility proxy."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from fork.extensions.hypura_oai_proxy import (
    _build_hypura_chat_payload,
    _openai_nonstream_response,
)


def test_chat_payload_forwards_messages_tools_and_options() -> None:
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

    payload = _build_hypura_chat_payload(body, "fallback")

    assert payload["model"] == "Agents A1 4B"
    assert payload["stream"] is True
    assert payload["messages"] == body["messages"]
    assert payload["tools"] == body["tools"]
    assert payload["options"]["temperature"] == 0
    assert payload["options"]["num_predict"] == 128


def test_chat_payload_rejects_missing_messages() -> None:
    with pytest.raises(HTTPException) as exc:
        _build_hypura_chat_payload({"model": "m"}, "fallback")
    assert exc.value.status_code == 400


def test_openai_nonstream_response_extracts_message_content() -> None:
    hypura = {
        "message": {"role": "assistant", "content": "Hello from Hypura"},
        "prompt_eval_count": 10,
        "eval_count": 7,
    }

    result = _openai_nonstream_response(hypura, "chatcmpl-test", 123, "Agents A1 4B")

    message = result["choices"][0]["message"]
    assert message["content"] == "Hello from Hypura"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 7,
        "total_tokens": 17,
    }


def test_openai_nonstream_response_handles_empty_message() -> None:
    result = _openai_nonstream_response({}, "chatcmpl-test", 123, "model")
    assert result["choices"][0]["message"]["content"] == ""
    assert result["usage"] is None
