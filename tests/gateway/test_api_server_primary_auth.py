"""A primary's lapsed login, carried by the API server's fallback runtime, stays with the route it fell back to."""
import pytest

from tests.gateway.test_api_server import _make_routing_adapter, _patch_create_agent_runtime



@pytest.mark.parametrize("route_extra, keeps", [
    ({"api_key": "sk-route"}, False),
    ({"base_url": "https://openrouter.ai/api/v1/other"}, False),
    ({}, True),
], ids=["own-key", "own-url", "unchanged"])
def test_a_model_route_that_changes_the_backups_account_drops_the_primary_login(monkeypatch, route_extra, keeps):
    """A model_routes entry applies its own api_key or base_url after fallback resolution. A refusal from
    that account is its own, not the primary's lapsed login; an entry that changes neither keeps it."""
    from agent.primary_auth_fallback import RUNTIME_KEY, PrimaryAuthFailure

    captured, agents = {}, []

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            agents.append(self)

    _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
    failure = PrimaryAuthFailure(provider="nous", error="expired", relogin="hermes portal", backup="openrouter")
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {
        "provider": "openrouter", "api_key": "sk-global", "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions", RUNTIME_KEY: failure})
    adapter = _make_routing_adapter({"alias": {"model": "global/model", **route_extra}})  # same model: no re-resolve
    monkeypatch.setattr(adapter, "_ensure_store", lambda: None)
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

    adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))

    assert (getattr(agents[0], "_primary_auth_failure", None) is failure) is keeps
