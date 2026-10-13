"""Primary redirect policy reaches real clients without changing transport ownership."""
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from agent.agent_init import _init_anthropic_client
from agent.agent_runtime_helpers import (
    _build_anthropic_client_from_runtime, _build_switched_client,
)
from agent.client_lifecycle import _swap_fallback_clients
from hermes_cli.config import atomic_config_replace
from run_agent import AIAgent


@pytest.fixture
def configure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def write(providers):
        atomic_config_replace(tmp_path / "config.yaml", {"providers": providers})

    return write


def _agent(provider="redirect-provider", base_url="https://redirect.example/v1", mode="chat_completions"):
    agent = AIAgent.__new__(AIAgent)
    agent.provider = provider
    agent.requested_provider = provider
    agent.model = "fixture-model"
    agent.base_url = base_url
    agent.api_key = "fixture-key"
    agent.api_mode = mode
    agent.quiet_mode = True
    agent._client_kwargs = {"api_key": agent.api_key, "base_url": base_url}
    return agent


@pytest.mark.parametrize("base_url", ["https://redirect.example/v1", "https://api.githubcopilot.com"])
@pytest.mark.parametrize("policy", [False, True])
def test_keepalive_wrapper_passes_policy(base_url, policy):
    with AIAgent._build_keepalive_http_client(base_url, follow_redirects=policy) as client:
        assert client.follow_redirects is policy


@pytest.mark.parametrize("provider,base_url", [
    ("redirect-provider", "https://redirect.example/v1"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta"),
])
@pytest.mark.parametrize("policy", [None, False, True])
def test_primary_openai_and_gemini_created_transport_policy(configure, provider, base_url, policy):
    configure({provider: {} if policy is None else {"follow_redirects": policy}})
    agent = _agent(provider, base_url)
    kwargs = dict(agent._client_kwargs)
    client = agent._create_openai_client(kwargs, reason="test", shared=True)
    try:
        transport = client._http if provider == "gemini" else client._client
        assert transport.follow_redirects is bool(policy)
        assert kwargs == agent._client_kwargs
        assert "http_client" not in kwargs
    finally:
        client.close()


@pytest.mark.parametrize("provider,base_url", [
    ("redirect-provider", "https://redirect.example/v1"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta"),
])
def test_caller_transport_wins_without_sdk_key_leak(configure, provider, base_url):
    configure({provider: {"follow_redirects": True}})
    agent = _agent(provider, base_url)
    with httpx.Client(follow_redirects=False) as owned:
        kwargs = {**agent._client_kwargs, "http_client": owned, "follow_redirects": True}
        client = agent._create_openai_client(kwargs, reason="test", shared=True)
        try:
            transport = client._http if provider == "gemini" else client._client
            assert transport is owned
            assert transport.follow_redirects is False
            assert kwargs["follow_redirects"] is True
        finally:
            client.close()


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize("provider,base_url", [
    ("redirect-provider", "https://redirect.example/v1"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta"),
])
def test_explicit_policy_fails_closed_if_transport_unavailable(configure, provider, base_url, policy):
    configure({provider: {"follow_redirects": policy}})
    agent = _agent(provider, base_url)
    with patch.object(agent, "_build_keepalive_http_client", return_value=None):
        with pytest.raises(RuntimeError, match="redirect policy"):
            agent._create_openai_client(agent._client_kwargs, reason="test", shared=True)


def test_provider_owned_client_not_replaced(configure, monkeypatch):
    configure({"redirect-provider": {"follow_redirects": True}})
    agent = _agent()
    owned = object()
    monkeypatch.setattr("agent.agent_runtime_helpers._provider_supplied_client", lambda *args: owned)
    with patch.object(agent, "_build_keepalive_http_client", side_effect=AssertionError("must not build")):
        assert agent._create_openai_client(agent._client_kwargs, reason="test", shared=True) is owned


@pytest.mark.parametrize("policy", [None, False, True])
@pytest.mark.parametrize("path", ["initial", "direct", "rebuild", "switch", "restore", "fallback"])
def test_direct_anthropic_lifecycle_policy(configure, policy, path):
    configure({"anthropic": {} if policy is None else {"follow_redirects": policy}})
    agent = _agent("anthropic", "https://api.anthropic.com", "anthropic_messages")
    agent._anthropic_api_key = agent.api_key = "sk-ant-api03-fixture"
    agent._anthropic_base_url = agent.base_url
    actions = {
        "initial": lambda: _init_anthropic_client(agent, agent.api_key, agent.base_url, None),
        "direct": lambda: setattr(agent, "_anthropic_client", agent._build_direct_anthropic_client(agent.api_key, agent.base_url)),
        "rebuild": agent._rebuild_anthropic_client,
        "switch": lambda: _build_switched_client(agent, "anthropic", agent.api_key, agent.base_url, agent.api_mode, "anthropic"),
        "restore": lambda: _build_anthropic_client_from_runtime(agent, {
            "anthropic_api_key": agent.api_key, "anthropic_base_url": agent.base_url,
            "is_anthropic_oauth": False,
        }),
        "fallback": lambda: _swap_fallback_clients(agent, SimpleNamespace(api_key=agent.api_key),
                                                 agent.provider, agent.model, agent.base_url, agent.api_mode),
    }
    actions[path]()
    try:
        assert agent._anthropic_client._client.follow_redirects is (True if policy is None else policy)
    finally:
        agent._anthropic_client.close()


@pytest.mark.parametrize("policy", [None, False, True])
def test_callable_anthropic_keeps_bare_httpx_default(configure, policy):
    configure({"anthropic": {} if policy is None else {"follow_redirects": policy}})
    agent = _agent("anthropic", "https://api.anthropic.com", "anthropic_messages")
    client = agent._build_direct_anthropic_client(lambda: "sk-ant-api03-fixture", agent.base_url)
    try:
        assert client._client.follow_redirects is bool(policy)
    finally:
        client.close()


@pytest.mark.parametrize("anthropic", [False, True])
def test_request_cache_evicts_on_policy_change(configure, anthropic):
    provider = "anthropic" if anthropic else "redirect-provider"
    agent = _agent(provider, "https://redirect.example/v1",
                   "anthropic_messages" if anthropic else "chat_completions")
    configure({provider: {"follow_redirects": True}})
    if anthropic:
        agent._anthropic_api_key = agent.api_key
        agent._anthropic_base_url = agent.base_url
        agent._try_refresh_anthropic_client_credentials = lambda: False
        create = agent._create_request_anthropic_client
        release = agent._close_request_anthropic_client
        cleanup = agent._close_cached_request_anthropic_client
    else:
        agent.client = agent._create_openai_client(agent._client_kwargs, reason="initial", shared=True)
        create = agent._create_request_openai_client
        release = agent._close_request_openai_client
        cleanup = agent._close_cached_request_openai_client
    try:
        first = create(reason="test")
        release(first, reason="request_complete")
        assert create(reason="test") is first
        release(first, reason="request_complete")
        configure({provider: {"follow_redirects": False}})
        second = create(reason="test")
        assert second is not first
        assert second._client.follow_redirects is False
        assert first.is_closed()
        release(second, reason="request_complete")
    finally:
        cleanup(reason="test")
        if not anthropic:
            agent.client.close()


def test_active_construction_route_wins_over_stale_identity(configure):
    configure({
        "old": {"base_url": "https://old.example/v1", "follow_redirects": True},
        "new": {"base_url": "https://new.example/v1", "follow_redirects": False},
    })
    agent = _agent("custom", "https://old.example/v1")
    agent.requested_provider = "custom:old"
    # A rebuild may happen before agent.base_url catches up to the incoming route.
    kwargs = {**agent._client_kwargs, "base_url": "https://new.example/v1"}
    client = agent._create_openai_client(kwargs, reason="rotation", shared=True)
    try:
        assert client._client.follow_redirects is False
    finally:
        client.close()
    client = agent._build_direct_anthropic_client("fixture", "https://new.example/v1")
    try:
        assert client._client.follow_redirects is False
    finally:
        client.close()


@pytest.mark.parametrize("anthropic", [False, True])
def test_credential_rotation_resolves_new_route_policy(configure, anthropic):
    configure({
        "old": {"base_url": "https://old.example/v1", "follow_redirects": True},
        "new": {"base_url": "https://new.example/v1", "follow_redirects": False},
    })
    agent = _agent("custom", "https://old.example/v1",
                   "anthropic_messages" if anthropic else "chat_completions")
    agent.requested_provider = "custom:old"
    if anthropic:
        agent._anthropic_api_key = agent.api_key
        agent._anthropic_base_url = agent.base_url
        agent._anthropic_client = agent._build_direct_anthropic_client(agent.api_key, agent.base_url)
    else:
        agent.client = agent._create_openai_client(agent._client_kwargs, reason="initial", shared=True)
    entry = SimpleNamespace(id="rotated", runtime_api_key="next-key",
                            runtime_base_url="https://new.example/v1")
    try:
        assert agent._swap_credential(entry)
        installed = agent._anthropic_client if anthropic else agent.client
        assert installed._client.follow_redirects is False
        assert agent.base_url == "https://new.example/v1"
        assert agent.requested_provider == "custom:old"
    finally:
        (agent._anthropic_client if anthropic else agent.client).close()


def test_unset_fallback_preserves_installed_client(configure):
    from openai import OpenAI
    configure({})
    agent = _agent()
    auxiliary = OpenAI(api_key="fixture-key", base_url=agent.base_url)
    try:
        _swap_fallback_clients(agent, auxiliary, agent.provider, agent.model, agent.base_url, agent.api_mode)
        assert agent.client is auxiliary
        assert auxiliary._client.follow_redirects is True
    finally:
        auxiliary.close()


def test_installed_openai_fallback_uses_primary_policy(configure):
    from openai import OpenAI
    configure({"redirect-provider": {"follow_redirects": False}})
    agent = _agent()
    source = lambda: "fixture-key"
    auxiliary = OpenAI(api_key=source, base_url=agent.base_url,
                       default_headers={"X-Provider": "kept"})
    try:
        assert auxiliary._client.follow_redirects is True
        _swap_fallback_clients(agent, auxiliary, agent.provider, agent.model, agent.base_url, agent.api_mode)
        assert agent.client is not auxiliary
        assert agent.client._client.follow_redirects is False
        assert agent.client.default_headers["X-Provider"] == "kept"
        assert agent._client_kwargs["api_key"] is source
        assert "http_client" not in agent._client_kwargs
        agent.client.close()
    finally:
        auxiliary.close()


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize("failure", ["transport", "sdk"])
def test_fallback_construction_failure_cannot_leave_unguarded_client(configure, policy, failure):
    from openai import OpenAI
    configure({"redirect-provider": {"follow_redirects": policy}})
    agent = _agent()
    auxiliary = OpenAI(api_key="fixture-key", base_url=agent.base_url)
    target = (patch.object(agent, "_build_keepalive_http_client", return_value=None)
              if failure == "transport" else
              patch("agent.process_bootstrap.OpenAI", side_effect=RuntimeError("SDK construction failed")))
    try:
        with target, pytest.raises(RuntimeError):
            _swap_fallback_clients(agent, auxiliary, agent.provider, agent.model, agent.base_url, agent.api_mode)
        assert agent.client is None
    finally:
        auxiliary.close()


@pytest.mark.parametrize("timeout", [None, 12])
def test_unset_fallback_rebuild_failure_preserves_legacy_client(configure, timeout):
    from openai import OpenAI
    configure({"redirect-provider": {} if timeout is None else {"request_timeout_seconds": timeout}})
    agent = _agent()
    auxiliary = OpenAI(api_key="fixture-key", base_url=agent.base_url)
    try:
        with patch.object(agent, "_build_keepalive_http_client", side_effect=RuntimeError("unavailable")) as build:
            _swap_fallback_clients(agent, auxiliary, agent.provider, agent.model, agent.base_url, agent.api_mode)
        assert agent.client is auxiliary
        assert build.call_count == (0 if timeout is None else 1)
    finally:
        auxiliary.close()


@pytest.mark.parametrize("provider,base_url", [
    ("redirect-provider", "https://redirect.example/v1"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta"),
])
@pytest.mark.parametrize("first_policy", [None, False, True])
def test_request_cache_uses_one_policy_snapshot(configure, provider, base_url, first_policy):
    configure({})
    agent = _agent(provider, base_url)
    agent.client = agent._create_openai_client(agent._client_kwargs, reason="initial", shared=True)
    try:
        with patch("hermes_cli.timeouts.get_provider_follow_redirects",
                   side_effect=[first_policy, not bool(first_policy)]) as resolve:
            first = agent._create_request_openai_client(reason="test")
            assert resolve.call_count == 1
            transport = first._http if provider == "gemini" else first._client
            assert transport.follow_redirects is bool(first_policy)
            assert agent._request_client_cache["key"][1] is first_policy
            agent._close_request_openai_client(first, reason="request_complete")
            second = agent._create_request_openai_client(reason="test")
            assert resolve.call_count == 2
            assert second is not first
            transport = second._http if provider == "gemini" else second._client
            assert transport.follow_redirects is not bool(first_policy)
            assert agent._request_client_cache["key"][1] is not bool(first_policy)
            agent._close_request_openai_client(second, reason="request_complete")
    finally:
        agent._close_cached_request_openai_client(reason="test")
        agent.client.close()


@pytest.mark.parametrize("kind", ["client", "transport"])
def test_provider_injection_precedes_redirect_policy(configure, monkeypatch, kind):
    import providers
    from providers.base import ProviderProfile
    configure({"redirect-provider": {"follow_redirects": True}})
    agent = _agent()
    with httpx.Client(follow_redirects=False) as owned:
        class Profile(ProviderProfile):
            def build_client_kwargs_extras(self, **context):
                return {"http_client": owned} if kind == "transport" else {}

            def create_client(self, **kwargs):
                assert "follow_redirects" not in kwargs
                assert "_follow_redirects" not in kwargs
                return owned if kind == "client" else None

        providers._discover_providers()
        monkeypatch.setitem(providers._REGISTRY, agent.provider, Profile(name=agent.provider))
        with patch.object(agent, "_build_keepalive_http_client", side_effect=AssertionError("must not build")):
            client = agent._create_openai_client(agent._client_kwargs, reason="test", shared=True,
                                                  _follow_redirects=True)
        try:
            assert (client if kind == "client" else client._client) is owned
            assert owned.follow_redirects is False
        finally:
            client.close()
