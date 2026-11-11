"""Auxiliary routes (compression, titles, vision, ...) honour routing_policy at resolve_provider_client."""

from types import SimpleNamespace

import pytest

from hermes_cli.routing_policy import RoutingPolicyError

_DENY = {"enabled": True, "deny": {"providers": ["blocked-provider"], "models": ["vendor-x/*"],
                                   "base_url_hosts": ["api.blocked.example"]}}


@pytest.fixture
def policy(monkeypatch):
    from hermes_cli import config

    state = {"policy": _DENY}
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {"routing_policy": state["policy"]})
    return state


def _fail(*_a, **_k):
    pytest.fail("no client may be built for a denied route")


@pytest.mark.parametrize("kwargs", [
    {"provider": "blocked-provider", "model": "m"},
    {"provider": "openrouter", "model": "vendor-x/m"},
    {"provider": "custom", "model": "m", "explicit_base_url": "https://api.blocked.example/v1"},
])
def test_denied_requested_route_is_refused_before_any_client_is_built(policy, monkeypatch, kwargs):
    from agent import auxiliary_client as aux

    monkeypatch.setattr(aux, "_resolve_provider_client_unchecked", _fail)
    with pytest.raises(RoutingPolicyError):
        aux.resolve_provider_client(**kwargs)


def test_require_explicit_refuses_auto_discovery(policy, monkeypatch):
    from agent import auxiliary_client as aux

    policy["policy"] = {"enabled": True, "require_explicit": True}
    monkeypatch.setattr(aux, "_resolve_provider_client_unchecked", _fail)
    with pytest.raises(RoutingPolicyError, match="explicit provider"):
        aux.resolve_provider_client("auto", model="m")


def test_auto_discovery_that_lands_on_a_denied_endpoint_is_refused(policy, monkeypatch):
    from agent import auxiliary_client as aux

    landed = SimpleNamespace(base_url="https://api.blocked.example/v1")
    monkeypatch.setattr(aux, "_resolve_provider_client_unchecked", lambda *a: (landed, "allowed-model"))
    with pytest.raises(RoutingPolicyError, match="base-url host"):
        aux.resolve_provider_client("auto")


def test_allowed_route_and_disabled_policy_pass_through(policy, monkeypatch):
    from agent import auxiliary_client as aux

    client = SimpleNamespace(base_url="https://allowed.example/v1")
    monkeypatch.setattr(aux, "_resolve_provider_client_unchecked", lambda *a: (client, "vendor-y/m"))
    assert aux.resolve_provider_client("openrouter", model="vendor-y/m") == (client, "vendor-y/m")
    policy["policy"] = {"enabled": False}
    monkeypatch.setattr(aux, "_resolve_provider_client_unchecked", lambda *a: (client, "vendor-x/m"))
    assert aux.resolve_provider_client("blocked-provider", model="vendor-x/m") == (client, "vendor-x/m")


def test_unresolvable_route_still_returns_none(policy, monkeypatch):
    from agent import auxiliary_client as aux

    monkeypatch.setattr(aux, "_resolve_provider_client_unchecked", lambda *a: (None, None))
    assert aux.resolve_provider_client("openrouter", model="m") == (None, None)


@pytest.mark.parametrize("label", ["blocked-provider", "fallback_chain[0](blocked-provider)"])
def test_auto_discovery_that_lands_on_a_denied_provider_is_refused(policy, monkeypatch, label):
    """``auto`` names no provider; the provider discovery picked (tagged on the client) is checked instead."""
    from agent import auxiliary_client as aux

    landed = SimpleNamespace(base_url="https://allowed.example/v1")
    monkeypatch.setattr(aux, "_resolve_auto_route", lambda **k: (landed, "allowed-model", label))
    monkeypatch.setattr(aux, "_route_client", lambda req, client, model: (client, model))
    with pytest.raises(RoutingPolicyError, match="denies provider 'blocked-provider'"):
        aux.resolve_provider_client("auto")


def test_auto_discovery_backend_tag_is_checked(policy, monkeypatch):
    from agent import auxiliary_client as aux

    landed = SimpleNamespace(base_url="https://allowed.example/v1", _hermes_aux_backend_provider="blocked-provider")
    monkeypatch.setattr(aux, "_resolve_provider_client_unchecked", lambda *a: (landed, "allowed-model"))
    with pytest.raises(RoutingPolicyError, match="denies provider"):
        aux.resolve_provider_client("")


def test_malformed_policy_refuses_auxiliary_routes(policy, monkeypatch):
    from agent import auxiliary_client as aux

    policy["policy"] = None
    monkeypatch.setattr(aux, "_resolve_provider_client_unchecked", _fail)
    with pytest.raises(RoutingPolicyError, match="invalid routing policy"):
        aux.resolve_provider_client("openrouter", model="m")
