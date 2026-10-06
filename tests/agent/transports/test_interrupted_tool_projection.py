"""Interrupted tool history is canonical; strict role bridges are request-local (#20154)."""

from copy import deepcopy

import pytest
from openai import OpenAI

from agent.auxiliary_wire import prepare_chat_messages
from agent.transports.chat_completions import ChatCompletionsTransport
from providers.base import ProviderProfile


def _history():
    return [
        {"role": "user", "content": "Read the file"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_read", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_read", "content": "file contents"},
        {"role": "user", "content": "Stop reading; summarize what you found"},
    ]


@pytest.mark.parametrize("entry", ["main", "auxiliary"])
@pytest.mark.parametrize("model,base_url,strict", [
    ("mistralai/mistral-small-4-119b-2603", "https://integrate.api.nvidia.com/v1", True),
    ("mistralai/magistral-small", "https://openrouter.ai/api/v1", True),
    ("private-alias", "https://api.mistral.ai/v1", True),
    ("gpt-4.1-mini", "https://api.openai.com/v1", False),
    ("google/gemini-2.5-flash", "https://openrouter.ai/api/v1", False),
    ("private-alias", "https://api.mistral.ai.example.com/v1", False),
])
def test_projection_is_destination_local_across_retries_and_fallback(entry, model, base_url, strict):
    history = _history()
    original = deepcopy(history)
    transport = ChatCompletionsTransport()

    def request_for(destination_model, destination_url):
        if entry == "main":
            return transport.build_kwargs(
                model=destination_model, messages=history, base_url=destination_url,
            )["messages"]
        with OpenAI(api_key="test-key", base_url=destination_url) as client:
            return prepare_chat_messages(client, {"model": destination_model, "messages": history})["messages"]

    wire = request_for(model, base_url)
    expected_roles = ["user", "assistant", "tool", "assistant", "user"] if strict else ["user", "assistant", "tool", "user"]
    assert [row["role"] for row in wire] == expected_roles
    assert wire[-1] == original[-1]
    assert wire[1:3] == original[1:3]
    assert request_for(model, base_url) == wire  # retry does not grow the cached prefix
    assert request_for("gpt-4.1-mini", "https://api.openai.com/v1") == original
    assert history == original


def test_provider_identity_covers_aliases_without_projecting_the_virtual_moa_route():
    history = _history()
    transport = ChatCompletionsTransport()
    mistral = transport.build_kwargs(
        model="private-alias", messages=history, provider_profile=ProviderProfile(name="mistral"),
    )["messages"]
    assert [row["role"] for row in mistral] == ["user", "assistant", "tool", "assistant", "user"]
    virtual = transport.convert_messages(history, model="mistral-team", provider_name="moa")
    assert virtual == history
    assert history == _history()


@pytest.mark.parametrize("provider", ["anthropic", "gemini", "responses"])
def test_native_provider_projection_preserves_canonical_history(provider):
    from agent.anthropic_message_convert import convert_messages_to_anthropic
    from agent.gemini_native_adapter import build_gemini_request
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    history = _history()
    original = deepcopy(history)
    if provider == "anthropic":
        _, request = convert_messages_to_anthropic(history)
        assert [m["role"] for m in request] == ["user", "assistant", "user"]
        assert [part["type"] for part in request[-1]["content"]] == ["tool_result", "text"]
        assert request[-1]["content"][-1]["text"] == history[-1]["content"]
    elif provider == "gemini":
        request = build_gemini_request(messages=history, tools=[], tool_choice=None)["contents"]
        # Native Gemini already owns its wire-only bridge; keep that policy intact.
        assert [m["role"] for m in request] == ["user", "model", "user", "model", "user"]
        assert "functionResponse" in request[2]["parts"][0]
        assert request[-1]["parts"][-1]["text"] == history[-1]["content"]
    else:
        request = _chat_messages_to_responses_input(history)
        assert [m.get("type", "message") for m in request] == ["message", "function_call", "function_call_output", "message"]
        assert request[-1]["role"] == "user"
    assert history == original
