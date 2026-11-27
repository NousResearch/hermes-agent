"""Fallback rebuilds retain the credential source, not its last token."""
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from agent.auxiliary_client import CodexAuxiliaryClient
from agent.client_lifecycle import _swap_fallback_clients


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_fallback_rebuild_preserves_rotating_and_static_credentials(mode, dynamic):
    token = ["first-token"]
    source = (lambda: token[0]) if dynamic else token[0]
    captured = []

    def respond(request):
        captured.append(request.headers.get("Authorization"))
        return httpx.Response(200, json={"id": "fixture", "choices": []})

    client = OpenAI(api_key=source, base_url="http://localhost:1234/v1")
    agent = SimpleNamespace()
    _swap_fallback_clients(agent, client, "fixture", "fixture", str(client.base_url), mode)
    rebuilt = OpenAI(**agent._client_kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        for value in ("first-token", "rotated-token"):
            token[0] = value
            rebuilt.chat.completions.create(model="fixture", messages=[])
        expected = ["Bearer first-token", "Bearer rotated-token" if dynamic else "Bearer first-token"]
        assert captured == expected
        assert agent.api_key is source
    finally:
        client.close()
        rebuilt.close()


def test_anthropic_fallback_keeps_callable_without_treating_it_as_oauth_text():
    source = lambda: "fixture-token"
    client = OpenAI(api_key=source, base_url="http://localhost:1234/v1")
    agent = SimpleNamespace()
    try:
        _swap_fallback_clients(agent, client, "anthropic", "fixture", str(client.base_url), "anthropic_messages")
        assert agent.api_key is source
        assert agent._anthropic_api_key is source
        assert agent._is_anthropic_oauth is False
        agent._anthropic_client.close()
    finally:
        client.close()


def test_codex_wrapper_fallback_keeps_entra_credential_source():
    """A CodexAuxiliaryClient (azure-foundry codex_responses fallback, Entra ID) must carry the
    rotating bearer: the wrapper's own .api_key mirrors the SDK's empty-string representation,
    so without _api_key_provider the swap hands the agent api_key="" and every call 401s (#135474)."""
    source = lambda: "fixture-entra"
    real = OpenAI(api_key=source, base_url="http://localhost:1234/v1")
    wrapper = CodexAuxiliaryClient(real, "fixture")
    agent = SimpleNamespace()
    try:
        _swap_fallback_clients(agent, wrapper, "fixture", "fixture", str(wrapper.base_url), "codex_responses")
        assert agent.api_key is source
        assert agent._client_kwargs["api_key"] is source
    finally:
        wrapper.close()


def test_codex_wrapper_fallback_keeps_static_string_credential():
    real = OpenAI(api_key="fixture", base_url="http://localhost:1234/v1")
    wrapper = CodexAuxiliaryClient(real, "fixture")
    agent = SimpleNamespace()
    try:
        _swap_fallback_clients(agent, wrapper, "fixture", "fixture", str(wrapper.base_url), "codex_responses")
        assert agent.api_key == "fixture"
        assert agent._client_kwargs["api_key"] == "fixture"
    finally:
        wrapper.close()
