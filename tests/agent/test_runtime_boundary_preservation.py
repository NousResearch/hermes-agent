"""Runtime configuration survives request ownership, fallback and rotation."""

from types import SimpleNamespace

import pytest

from agent.client_lifecycle import ClientLifecycleMixin
from agent.runtime_bundle import ResolvedRuntime, build_client_bundle


class Agent(ClientLifecycleMixin):
    def _try_refresh_anthropic_client_credentials(self):
        return False


def runtime(**updates):
    return ResolvedRuntime.from_mapping({
        "provider": "custom", "model": "model", "api_mode": "anthropic_messages",
        "api_key": "old-key", "base_url": "https://route.invalid/v1",
        "extra_headers": {"X-Route": "one"}, "default_query": {"route": "one"},
        "timeout": 23, "ssl_ca_cert": "route.pem", **updates,
    })


@pytest.fixture
def wire(monkeypatch):
    def build(api_key, base_url, **kwargs):
        return SimpleNamespace(api_key=api_key, base_url=base_url, kwargs=kwargs, close=lambda: None)
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", build)
    return build


def test_request_client_and_reuse_key_preserve_complete_runtime(wire):
    rt = runtime()
    agent = Agent()
    agent.install_runtime(build_client_bundle(rt))
    request = agent._create_request_anthropic_client(reason="test")
    assert request.kwargs["default_headers"]["X-Route"] == "one"
    assert request.kwargs["default_query"] == {"route": "one"}
    assert request.kwargs["ssl_ca_cert"] == "route.pem"
    assert request.kwargs["timeout"] == 23
    agent._close_request_anthropic_client(request, reason="request_complete")
    old_key = agent._request_anthropic_client_key()
    agent._resolved_runtime = rt.with_updates(extra_headers={"X-Route": "two"})
    assert agent._request_anthropic_client_key() != old_key
    replacement = agent._create_request_anthropic_client(reason="test")
    assert replacement is not request
    assert replacement.kwargs["default_headers"]["X-Route"] == "two"
    assert rt.extra_headers["X-Route"] == "one"


def test_rotation_preserves_runtime_and_build_failure_is_atomic(wire, monkeypatch):
    agent = Agent()
    agent.install_runtime(build_client_bundle(runtime()))
    entry = SimpleNamespace(id="second", runtime_api_key="new-key", runtime_base_url=agent.base_url)
    agent._swap_credential(entry)
    assert agent._resolved_runtime.api_key == agent.api_key == "new-key"
    assert agent._anthropic_client.kwargs["default_headers"]["X-Route"] == "one"
    assert agent._anthropic_client.kwargs["default_query"] == {"route": "one"}
    installed = agent._resolved_runtime
    old_client = agent._anthropic_client
    def fail(*args, **kwargs):
        raise RuntimeError("build failed")
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", fail)
    entry.id, entry.runtime_api_key = "third", "bad-key"
    with pytest.raises(RuntimeError, match="build failed"):
        agent._swap_credential(entry)
    assert agent._resolved_runtime is installed
    assert agent._anthropic_client is old_client
    assert agent.api_key == "new-key"
    assert agent._credential_pool_entry_id == "second"


def test_fallback_rebuild_keeps_resolver_runtime():
    from agent.chat_completion_helpers import _swap_fallback_clients
    rt = runtime(api_mode="chat_completions")
    client = SimpleNamespace(
        api_key=rt.api_key, base_url=rt.base_url, _hermes_resolved_runtime=rt,
        _custom_headers=dict(rt.extra_headers), default_query={"route": "one"},
        timeout=23, ssl_ca_cert="route.pem",
    )
    agent = Agent()
    _swap_fallback_clients(agent, client, rt.provider, rt.model, rt.base_url, rt.api_mode)
    assert agent.client is client
    assert agent._resolved_runtime.ssl_ca_cert == rt.ssl_ca_cert
    assert agent._client_kwargs["ssl_ca_cert"] == rt.ssl_ca_cert
    assert agent._client_kwargs["default_query"] == {"route": "one"}
    assert agent._client_kwargs["default_headers"]["X-Route"] == "one"
    assert agent._client_kwargs["timeout"] == 23


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
def test_endpoint_rotation_does_not_leak_previous_route_settings(wire, monkeypatch, api_mode):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"custom_providers": [{
        "name": "next", "base_url": "https://next.invalid/v1", "extra_headers": {"X-Next": "next"},
    }]})
    agent = Agent()
    agent._create_openai_client = lambda kwargs, **kw: SimpleNamespace(kwargs=kwargs, close=lambda: None)
    rt = runtime(api_mode=api_mode, extra_headers={"X-Secret": "old-route-secret"})
    agent.install_runtime(build_client_bundle(rt, openai_builder=lambda kwargs: agent._create_openai_client(kwargs)))
    agent._swap_credential(SimpleNamespace(id="next", runtime_api_key="next-key", runtime_base_url="https://next.invalid/v1"))
    resolved = agent._resolved_runtime
    assert resolved.api_key == "next-key"
    assert resolved.base_url == "https://next.invalid/v1"
    assert resolved.extra_headers == {"X-Next": "next"}
    assert resolved.ssl_ca_cert is None
    assert not resolved.get("default_query")
    client = agent._anthropic_client if api_mode == "anthropic_messages" else agent.client
    assert client.kwargs["default_headers"] == {"X-Next": "next"}
    assert "ssl_ca_cert" not in client.kwargs
    assert not client.kwargs.get("default_query")
