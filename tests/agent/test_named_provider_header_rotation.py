"""Named custom headers stay on their own credential and endpoint through rotation."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import OpenAI

from run_agent import AIAgent

URL = "https://team.example.invalid/v1"
HEADERS = {"X-Account": "team-one-secret"}


def _entry(key, url=URL, provider="custom:team-one"):
    return SimpleNamespace(id=key, provider=provider, runtime_api_key=key,
                           runtime_base_url=url, access_token="", base_url=url)


@pytest.mark.parametrize("stream", [False, True])
def test_named_openai_rotation_reaches_shared_and_request_wire(stream):
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        agent = AIAgent(api_key="old", base_url=URL, model="fixture", provider="custom",
                        requested_provider="custom:team-one", extra_headers=HEADERS,
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._credential_pool = SimpleNamespace(provider="custom:team-one")
    seen = []
    def respond(req):
        seen.append(req)
        if stream:
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text='data: {"choices":[{"delta":{"content":"ok"},"index":0}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"id":"x", "object":"chat.completion", "created":1,
                                         "model":"fixture", "choices":[]})
    def build(kwargs, **_):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    with (patch.object(agent, "_create_openai_client", side_effect=build),
          patch.object(agent, "_retire_shared_openai_client")):
        assert agent._swap_credential(_entry("new"))
        assert agent._client_kwargs["default_headers"]["X-Account"] == HEADERS["X-Account"]
        request_client = agent._create_request_openai_client(reason="rotated-wire")
        for client in (agent.client, request_client):
            result = client.chat.completions.create(model="fixture", messages=[{"role":"user","content":"hi"}], stream=stream)
            if stream:
                list(result)
        agent._close_request_openai_client(request_client, reason="test_complete")
        agent.client.close()
    assert len(seen) == 2
    assert all(req.headers["x-account"] == HEADERS["X-Account"] and
               req.headers["authorization"] == "Bearer new" for req in seen)


def test_named_openai_rotation_never_carries_headers_to_other_credential_or_endpoint():
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        agent = AIAgent(api_key="old", base_url=URL, model="fixture", provider="custom",
                        requested_provider="custom:team-one", extra_headers=HEADERS,
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._credential_pool = SimpleNamespace(provider="custom:team-one")
    with patch.object(agent, "_replace_primary_openai_client", return_value=True):
        agent._swap_credential(_entry("outsider", provider="custom:team-two"))
        assert "X-Account" not in agent._client_kwargs.get("default_headers", {})
        agent._swap_credential(_entry("other-url", url="https://other.example.invalid/v1"))
    assert "X-Account" not in agent._client_kwargs.get("default_headers", {})


@pytest.mark.parametrize("requested_provider,fallback_active", [("custom:team-one", False), ("custom", True)])
def test_openai_header_rebuild_does_not_resolve_another_providers_headers_by_shared_url(
        requested_provider, fallback_active):
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        agent = AIAgent(api_key="other-account", base_url=URL, model="fixture", provider="custom",
                        requested_provider=requested_provider, quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    agent._client_kwargs = {"api_key": "other-account", "base_url": URL}
    agent._fallback_activated = fallback_active
    def from_unrelated_provider(kwargs, _url):
        kwargs["default_headers"] = HEADERS
    with patch("hermes_cli.config.apply_custom_provider_extra_headers_to_client_kwargs",
               side_effect=from_unrelated_provider) as url_lookup:
        agent._apply_client_headers_for_base_url(URL, apply_user_headers=False)
    url_lookup.assert_not_called()
    assert "X-Account" not in agent._client_kwargs.get("default_headers", {})


def test_fallback_anthropic_does_not_resolve_another_providers_headers_by_shared_url(monkeypatch):
    from agent.client_lifecycle import _swap_fallback_clients

    captured = []
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client",
                        lambda key, url, **kwargs: captured.append(kwargs["extra_headers"]) or MagicMock())
    fallback = SimpleNamespace(api_key="other-account", _custom_headers={})
    agent = SimpleNamespace(model="fixture", _client_kwargs={})
    _swap_fallback_clients(agent, fallback, "custom", "fixture", URL, "anthropic_messages")
    assert captured == [{}]


def test_named_anthropic_rotation_keeps_headers_for_shared_and_request_client(monkeypatch):
    built = []
    def build(key, url, **kwargs):
        built.append((key, url, kwargs.get("extra_headers")))
        return MagicMock()
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", build)
    agent = SimpleNamespace(provider="custom", requested_provider="custom:team-one", model="fixture",
                            api_mode="anthropic_messages", api_key="old", base_url=URL,
                            _anthropic_api_key="old", _anthropic_base_url=URL,
                            _anthropic_client=MagicMock(), _resolved_header_api_key="old",
                            _resolved_extra_headers=HEADERS, _resolved_header_base_url=URL,
                            _credential_pool=SimpleNamespace(provider="custom:team-one"),
                            _fallback_activated=False, _client_kwargs={})
    from types import MethodType
    agent._build_direct_anthropic_client = MethodType(AIAgent._build_direct_anthropic_client, agent)
    agent._anthropic_oauth_flag = MethodType(AIAgent._anthropic_oauth_flag, agent)
    assert AIAgent._swap_credential(agent, _entry("new"))
    AIAgent._build_anthropic_client_for_key(agent, ("direct", "new", URL, None, False))
    assert built == [("new", URL, HEADERS), ("new", URL, HEADERS)]
    AIAgent._build_anthropic_client_for_key(agent, ("direct", "outsider", URL, None, False))
    AIAgent._build_anthropic_client_for_key(agent, ("direct", "new", "https://other.invalid/v1", None, False))
    assert built[-2][2] == built[-1][2] == {}
