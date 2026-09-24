"""Fallback rebuilds retain the credential source, not its last token."""
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

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
    source = lambda: "fixture-token"  # noqa: E731
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


def test_fallback_rebuild_keeps_the_tenant_query():
    """A query-bearing fallback URL (…/t?team=a) lives in the client as base_url + _custom_query.
    _client_kwargs must carry both, or every rebuild — and every child built from them — calls
    the tenant-less URL (review 6 of the OAuth-proxy work, finding 1)."""
    seen = []

    def respond(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"id": "fixture", "choices": []})

    client = OpenAI(api_key="k", base_url="https://relay.example.com/t", default_query={"team": "a"})
    agent = SimpleNamespace()
    _swap_fallback_clients(agent, client, "relay", "fixture", str(client.base_url), "chat_completions")
    assert agent._client_kwargs["default_query"] == {"team": "a"}
    rebuilt = OpenAI(**agent._client_kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        rebuilt.chat.completions.create(model="fixture", messages=[])
        assert seen == ["https://relay.example.com/t/chat/completions?team=a"]
    finally:
        client.close()
        rebuilt.close()


def test_fallback_without_a_query_adds_no_default_query():
    client = OpenAI(api_key="k", base_url="https://relay.example.com/t")
    agent = SimpleNamespace()
    try:
        _swap_fallback_clients(agent, client, "relay", "fixture", str(client.base_url), "chat_completions")
        assert "default_query" not in agent._client_kwargs
    finally:
        client.close()
