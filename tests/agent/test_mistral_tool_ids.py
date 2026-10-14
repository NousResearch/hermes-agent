"""Mistral rejects any tool id that is not exactly 9 alphanumeric characters.

A split ``tool_call`` batch mints ``call_`` + 12 hex via ``deterministic_call_id``.
Replaying that id to Mistral, including Mistral models on Ollama Cloud, is an
HTTP 400. The wire copy must map both sides of the pair to the same 9-character
id and leave stored history alone.
"""

from __future__ import annotations

import re

import pytest

from agent.message_sanitization import (
    mistral_compliant_tool_id,
    rewrite_mistral_tool_call_ids,
    route_requires_mistral_tool_ids,
)
from agent.transports.chat_completions import ChatCompletionsTransport

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
            ],
        },
        {"role": "tool", "tool_call_id": _NATIVE_ID, "content": "windows"},
        {"role": "tool", "tool_call_id": _BAD_ID, "content": "capture"},
    ]


def test_split_id_maps_to_one_stable_mistral_id():
    assert mistral_compliant_tool_id(_BAD_ID) == "uzvbkiH61"
    assert mistral_compliant_tool_id(_NATIVE_ID) == _NATIVE_ID
    assert mistral_compliant_tool_id(_BAD_ID) == mistral_compliant_tool_id(_BAD_ID)


@pytest.mark.parametrize(
    ("provider", "model", "base_url", "required"),
    [
        ("ollama-cloud", "mistral-large-4", "https://ollama.com/v1", True),
        ("ollama-cloud", "devstral-2", "https://ollama.com/v1", True),
        ("mistral", "codestral-latest", "https://api.mistral.ai/v1", True),
        ("custom", "pixtral-large", "https://example.invalid/v1", True),
        ("ollama-cloud", "glm-5.3-flash", "https://ollama.com/v1", False),
        ("openrouter", "anthropic/claude-sonnet-4", "https://openrouter.ai/api/v1", False),
    ],
)
def test_route_gate(provider, model, base_url, required):
    assert route_requires_mistral_tool_ids(provider, model, base_url) is required


def test_rewrite_pairs_both_sides_and_leaves_history():
    messages = _pair()
    rewritten = rewrite_mistral_tool_call_ids(
        messages, provider="ollama-cloud", model="mistral-large-4", base_url="https://ollama.com/v1",
    )
    assistant_ids = [tc["id"] for tc in rewritten[0]["tool_calls"]]
    result_ids = [rewritten[1]["tool_call_id"], rewritten[2]["tool_call_id"]]

    assert assistant_ids[0] == _NATIVE_ID
    assert _MISTRAL_ID.fullmatch(assistant_ids[1])
    assert assistant_ids[1] == "uzvbkiH61"
    assert rewritten[0]["tool_calls"][1]["call_id"] == assistant_ids[1]
    assert result_ids == assistant_ids
    assert messages[0]["tool_calls"][1]["id"] == _BAD_ID
    assert messages[2]["tool_call_id"] == _BAD_ID


def test_non_mistral_route_is_unchanged():
    messages = _pair()
    rewritten = rewrite_mistral_tool_call_ids(
        messages, provider="ollama-cloud", model="glm-5.3-flash", base_url="https://ollama.com/v1",
    )
    assert rewritten is messages


def test_ollama_cloud_profile_sends_the_rewritten_id():
    import model_tools  # registers provider profiles
    import providers

    profile = providers.get_provider_profile("ollama-cloud")
    messages = _pair()
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="mistral-large-4",
        messages=messages,
        tools=None,
        provider_profile=profile,
        base_url="https://ollama.com/v1",
    )
    sent_ids = [tc["id"] for tc in kwargs["messages"][0]["tool_calls"]]
    assert sent_ids[1] == "uzvbkiH61"
    assert "call_id" not in kwargs["messages"][0]["tool_calls"][1]
    assert kwargs["messages"][2]["tool_call_id"] == sent_ids[1]
    assert messages[2]["tool_call_id"] == _BAD_ID
