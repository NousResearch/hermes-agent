"""A local session's launch-policy route carries the provider's capabilities to the agent.

Local (Desktop / CLI) sessions resolve through ``_resolve_policy_agent_runtime``. When it dropped
``capabilities``, a route declaring ``anthropic_oauth_proxy`` lost its wire policy on every turn:
x-api-key instead of the Bearer, no Claude Code identity, and the relay's upstream answered 429.
"""
from types import SimpleNamespace

CAPS = {"anthropic_oauth_proxy": True, "ignore-me": "not-a-bool"}
RUNTIME = {"api_key": "relay-key", "base_url": "https://relay.example.com", "provider": "custom",
           "requested_provider": "relay", "api_mode": "anthropic_messages", "capabilities": CAPS}


def test_runtime_agent_kwargs_forward_filtered_capabilities():
    from gateway.run import _runtime_agent_kwargs
    assert _runtime_agent_kwargs(RUNTIME)["capabilities"] == {"anthropic_oauth_proxy": True}


def _policy_runtime(monkeypatch, fallback_entry=None):
    import gateway.session_authorities as authorities
    from gateway import session_policy
    import gateway.session_worker_construct as construct
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    monkeypatch.setattr(authorities, "active_authority", lambda runner: None)
    monkeypatch.setattr(session_policy, "route_launch_key", lambda *a, **k: None)
    monkeypatch.setattr(construct, "route_runtime", lambda *a, **k: (dict(RUNTIME), fallback_entry))
    policy = SimpleNamespace(effective=lambda authority: ({"model": {"provider": "relay"}}, "relay-model"))
    return _resolve_policy_agent_runtime(SimpleNamespace(), policy)


def test_policy_route_keeps_capabilities(monkeypatch):
    model, kwargs = _policy_runtime(monkeypatch)
    assert model == "relay-model"
    assert kwargs["capabilities"] == {"anthropic_oauth_proxy": True}


def test_policy_fallback_route_keeps_its_own_capabilities(monkeypatch):
    model, kwargs = _policy_runtime(monkeypatch, {"model": "fallback-model", "provider": "relay"})
    assert model == "fallback-model"
    assert kwargs["capabilities"] == {"anthropic_oauth_proxy": True}
