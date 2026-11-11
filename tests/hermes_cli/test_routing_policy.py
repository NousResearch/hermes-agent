"""routing_policy: opt-in deny lists / require_explicit, checked where model routes are resolved."""

import pytest

from hermes_cli.routing_policy import (
    RoutingPolicyError,
    check_requested_route,
    check_route,
    current_routing_policy,
    validate_routing_policy,
)

_DENY = {"enabled": True, "deny": {"providers": ["blocked-provider"], "models": ["vendor-x/*", "family-x-*"],
                                   "base_url_hosts": ["api.blocked.example"]}}


def test_absent_or_disabled_policy_allows_every_route():
    assert validate_routing_policy(None)["enabled"] is False
    check_route(None, provider="blocked-provider", model="vendor-x/m", base_url="https://api.blocked.example/v1")
    check_route({"enabled": False, **{k: v for k, v in _DENY.items() if k != "enabled"}},
                provider="blocked-provider", model="vendor-x/m", base_url="https://api.blocked.example/v1")


@pytest.mark.parametrize("route", [
    {"provider": "blocked-provider", "model": "ok", "base_url": "https://allowed.example/v1"},
    {"provider": "openrouter", "model": "FAMILY-X-2:free", "base_url": "https://allowed.example/v1"},
    {"provider": "openrouter", "model": "openrouter:vendor-x/m-2", "base_url": "https://allowed.example/v1"},
    {"provider": "openrouter", "model": "ok", "base_url": "https://eu.api.blocked.example/v1"},
])
def test_denied_provider_model_or_host_is_refused(route):
    with pytest.raises(RoutingPolicyError):
        check_route(_DENY, **route)


def test_allowed_route_passes():
    check_route(_DENY, provider="openrouter", model="vendor-y/m", base_url="https://notblocked.example/v1")


@pytest.mark.parametrize("provider, model, match", [("auto", "m", "explicit provider"), ("", "m", "explicit provider"),
                                                     ("openrouter", "", "explicit model")])
def test_require_explicit_refuses_discovery_and_silent_defaults(provider, model, match):
    with pytest.raises(RoutingPolicyError, match=match):
        check_requested_route({"enabled": True, "require_explicit": True}, requested_provider=provider, model=model)


@pytest.mark.parametrize("policy", [
    "yes", {"enabled": "true"}, {"unknown": 1}, {"enabled": True, "deny": {"hosts": []}},
    {"enabled": True, "deny": {"models": "vendor-x/*"}}, {"enabled": True, "deny": {"base_url_hosts": ["https://u:p@h"]}},
])
def test_malformed_policy_is_rejected_not_ignored(policy):
    with pytest.raises(RoutingPolicyError, match="invalid routing policy"):
        validate_routing_policy(policy)


def test_null_policy_in_config_is_rejected(monkeypatch):
    from hermes_cli import config

    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {"routing_policy": None})
    with pytest.raises(RoutingPolicyError, match="invalid routing policy"):
        current_routing_policy()


@pytest.mark.parametrize("raw", [None, "yes", {"enabled": True, "deny": {"hosts": []}}])
def test_null_or_malformed_policy_fails_every_runtime_resolution(monkeypatch, raw):
    """Fail closed: a broken routing_policy block stops resolution instead of silently disabling the policy."""
    from hermes_cli import config
    from hermes_cli import runtime_provider

    monkeypatch.setattr(runtime_provider, "load_config", lambda: {"model": {"provider": "openrouter", "default": "m"}})
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {"routing_policy": raw})
    monkeypatch.setattr(runtime_provider, "_ladder_rungs",
                        lambda *a, **k: pytest.fail("credential discovery must not run"))

    with pytest.raises(RoutingPolicyError, match="invalid routing policy"):
        runtime_provider.resolve_runtime_provider(requested="openrouter")


def test_runtime_resolver_refuses_implicit_route_before_credential_discovery(monkeypatch):
    from hermes_cli import config
    from hermes_cli import runtime_provider

    monkeypatch.setattr(runtime_provider, "load_config", lambda: {"model": {"provider": "", "default": ""}})
    monkeypatch.setattr(config, "read_raw_config_readonly",
                        lambda: {"routing_policy": {"enabled": True, "require_explicit": True}})
    monkeypatch.setattr(runtime_provider, "_ladder_rungs",
                        lambda *a, **k: pytest.fail("credential discovery must not run"))

    with pytest.raises(RoutingPolicyError, match="explicit provider"):
        runtime_provider.resolve_runtime_provider()


def test_runtime_resolver_refuses_a_denied_requested_provider_before_discovery(monkeypatch):
    from hermes_cli import config
    from hermes_cli import runtime_provider

    monkeypatch.setattr(runtime_provider, "load_config", lambda: {"model": {"provider": "openrouter", "default": "m"}})
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {"routing_policy": _DENY})
    monkeypatch.setattr(runtime_provider, "_ladder_rungs",
                        lambda *a, **k: pytest.fail("credential discovery must not run"))

    with pytest.raises(RoutingPolicyError, match="denies provider"):
        runtime_provider.resolve_runtime_provider(requested="blocked-provider")


def test_runtime_resolver_refuses_the_resolved_endpoint(monkeypatch):
    from hermes_cli import config
    from hermes_cli import runtime_provider

    monkeypatch.setattr(runtime_provider, "load_config", lambda: {"model": {"provider": "custom", "default": "m"}})
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {"routing_policy": _DENY})
    monkeypatch.setattr(runtime_provider, "_ladder_rungs", lambda *a, **k: iter(({
        "provider": "custom", "base_url": "https://api.blocked.example/v1", "api_key": "test", "api_mode": "chat_completions",
    },)))

    with pytest.raises(RoutingPolicyError, match="base-url host"):
        runtime_provider.resolve_runtime_provider()


def test_another_providers_configured_host_does_not_block_an_allowed_explicit_route(monkeypatch):
    """model.base_url belongs to model.provider; an explicitly selected other provider never uses it."""
    from hermes_cli import config
    from hermes_cli import runtime_provider

    model_cfg = {"provider": "zai", "default": "m", "base_url": "https://api.blocked.example/v1"}
    monkeypatch.setattr(runtime_provider, "load_config", lambda: {"model": dict(model_cfg)})
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {"routing_policy": _DENY})
    monkeypatch.setattr(runtime_provider, "_ladder_rungs", lambda *a, **k: iter(({
        "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "test",
        "api_mode": "chat_completions",
    },)))

    runtime = runtime_provider.resolve_runtime_provider(requested="openrouter")
    assert runtime["provider"] == "openrouter"

    monkeypatch.setattr(runtime_provider, "_ladder_rungs",
                        lambda *a, **k: pytest.fail("credential discovery must not run"))
    with pytest.raises(RoutingPolicyError, match="base-url host"):
        runtime_provider.resolve_runtime_provider(requested="zai")  # the provider that owns the denied host
    with pytest.raises(RoutingPolicyError, match="base-url host"):
        runtime_provider.resolve_runtime_provider(requested="openrouter",
                                                  explicit_base_url="https://eu.api.blocked.example/v1")


def test_resolution_fallback_never_steps_past_a_denied_entry(monkeypatch):
    """A primary auth failure walks the fallback chain; a policy-denied entry stops it."""
    from hermes_cli import config
    from hermes_cli import runtime_provider
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {"routing_policy": _DENY})
    calls = []

    def fake_resolve(**kwargs):
        calls.append(kwargs.get("requested"))
        if kwargs.get("requested") in (None, "openrouter"):
            raise AuthError("no key")
        raise RoutingPolicyError("routing policy denies provider 'blocked-provider'", code="denied_provider")

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", fake_resolve)
    cfg = {"fallback_providers": [{"provider": "blocked-provider", "model": "m"}, {"provider": "other", "model": "m"}]}
    with pytest.raises(RoutingPolicyError):
        runtime_provider.resolve_runtime_with_fallback(cfg, requested="openrouter")
    assert "other" not in calls


def test_policy_follows_the_bound_profile_a_to_b_to_a(tmp_path, monkeypatch):
    from agent import secret_scope
    from gateway.run import _profile_runtime_scope

    root = tmp_path / "hermes"
    beta = root / "profiles" / "beta"
    beta.mkdir(parents=True)
    (root / "config.yaml").write_text("routing_policy:\n  enabled: true\n", encoding="utf-8")
    (beta / "config.yaml").write_text("routing_policy:\n  enabled: true\n  deny:\n    models: ['vendor-x/*']\n",
                                      encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        for home, denied in ((root, False), (beta, True), (root, False)):
            with _profile_runtime_scope(home, {}):
                if denied:
                    with pytest.raises(RoutingPolicyError, match="selected model"):
                        check_route(current_routing_policy(), provider="openrouter", model="vendor-x/m", base_url="")
                else:
                    check_route(current_routing_policy(), provider="openrouter", model="vendor-x/m", base_url="")
    finally:
        secret_scope.set_multiplex_active(previous)
