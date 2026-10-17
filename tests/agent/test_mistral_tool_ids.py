"""Mistral rejects any tool id that is not exactly 9 alphanumeric characters.

A split ``tool_call`` batch mints ``call_`` + 12 hex via ``deterministic_call_id``.
Replaying that id to Mistral, including Mistral models on Ollama Cloud, is an
HTTP 400. The wire copy must map both sides of the pair to the same 9-character
id and leave stored history alone.
"""

from __future__ import annotations

import re

import pytest

from agent.transports.chat_completions import (
    ChatCompletionsTransport,
    _mistral_tool_id,
    _route_requires_mistral_tool_ids,
)

_BAD_ID = "call_04c3699a572b"
_NATIVE_ID = "c9Abb5ZtX"
_MISTRAL_ID = re.compile(r"^[A-Za-z0-9]{9}$")


def _pair():
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": _NATIVE_ID, "type": "function", "function": {"name": "tool_call", "arguments": "{}"}},
                {"id": _BAD_ID, "call_id": _BAD_ID, "type": "function",
                 "function": {"name": "tool_call", "arguments": "{}"}},
                # Composite Responses-bridge id (uniquify_tool_call_ids writes these); its result
                # carries only the call half (tool_dispatch_helpers._normalize_tool_call_id).
                {"id": "call_abc|fc_123", "call_id": "call_abc", "type": "function",
                 "function": {"name": "tool_call", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": _NATIVE_ID, "content": "windows"},
        {"role": "tool", "tool_call_id": _BAD_ID, "content": "capture"},
        {"role": "tool", "tool_call_id": "call_abc", "content": "bridge"},
    ]


def _wire(model, base_url="https://ollama.com/v1"):
    import model_tools  # registers provider profiles
    import providers

    messages = _pair()
    kwargs = ChatCompletionsTransport().build_kwargs(
        model=model, messages=messages, tools=None,
        provider_profile=providers.get_provider_profile("ollama-cloud"), base_url=base_url,
    )
    return messages, kwargs["messages"]


@pytest.mark.parametrize(
    ("provider", "model", "base_url", "required"),
    [
        ("ollama-cloud", "mistral-large-4", "https://ollama.com/v1", True),
        ("ollama-cloud", "devstral-2", "https://ollama.com/v1", True),
        ("mistral", "codestral-latest", "https://api.mistral.ai/v1", True),
        ("custom", "anything", "https://codestral.mistral.ai/v1", True),
        ("custom", "pixtral-large", "https://example.invalid/v1", True),
        ("custom", "magistral-small", "http://localhost:8000/v1", True),
        ("custom", "mixtral-8x22b", "http://localhost:8000/v1", True),
        ("ollama-cloud", "glm-5.3-flash", "https://ollama.com/v1", False),
        ("openrouter", "anthropic/claude-sonnet-4", "https://openrouter.ai/api/v1", False),
    ],
)
def test_route_gate(provider, model, base_url, required):
    assert _route_requires_mistral_tool_ids(provider, model, base_url) is required


def test_mistral_wire_pairs_every_call_with_its_result_and_leaves_history():
    assert _mistral_tool_id(_NATIVE_ID) == _NATIVE_ID
    assert _mistral_tool_id(_BAD_ID) == "uzvbkiH61"

    messages, sent = _wire("mistral-large-4")
    call_ids = [tc["id"] for tc in sent[0]["tool_calls"]]
    result_ids = [m["tool_call_id"] for m in sent[1:]]

    assert call_ids[:2] == [_NATIVE_ID, "uzvbkiH61"]
    assert all(_MISTRAL_ID.fullmatch(i) for i in call_ids)
    assert result_ids == call_ids
    assert all("call_id" not in tc for tc in sent[0]["tool_calls"])
    assert messages[0]["tool_calls"][1]["id"] == _BAD_ID
    assert messages[0]["tool_calls"][2]["id"] == "call_abc|fc_123"
    assert messages[2]["tool_call_id"] == _BAD_ID


def test_non_mistral_route_keeps_ids():
    _, sent = _wire("glm-5.3-flash")
    assert [tc["id"] for tc in sent[0]["tool_calls"]] == [_NATIVE_ID, _BAD_ID, "call_abc|fc_123"]
    assert [m["tool_call_id"] for m in sent[1:]] == [_NATIVE_ID, _BAD_ID, "call_abc"]
