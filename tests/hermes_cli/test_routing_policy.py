"""Behavioral coverage for terminal model-route policy enforcement."""

import pytest


def test_default_config_exposes_disabled_routing_policy():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["routing_policy"] == {
        "enabled": False,
        "require_explicit": False,
        "deny": {"providers": [], "models": [], "base_url_hosts": []},
    }


def test_disabled_policy_allows_every_route():
    from hermes_cli.routing_policy import check_route

    check_route(
        {"enabled": False},
        provider="openrouter",
        model="z-ai/glm-5.2",
        base_url="https://api.z.ai/v1",
    )


def test_enabled_policy_requires_explicit_route_before_credential_discovery():
    from hermes_cli.routing_policy import RoutingPolicyError, check_requested_route

    with pytest.raises(RoutingPolicyError, match="explicit provider"):
        check_requested_route(
            {"enabled": True, "require_explicit": True},
            requested_provider="auto",
            model="",
        )


def test_enabled_policy_denies_provider_model_and_base_url_host():
    from hermes_cli.routing_policy import RoutingPolicyError, check_route

    policy = {
        "enabled": True,
        "deny": {
            "providers": ["zai"],
            "models": ["glm-*", "z-ai/*"],
            "base_url_hosts": ["api.z.ai"],
        },
    }
    for route in (
        {"provider": "zai", "model": "allowed", "base_url": "https://allowed.example/v1"},
        {"provider": "openrouter", "model": "GLM-5.2:free", "base_url": "https://allowed.example/v1"},
        {"provider": "openrouter", "model": "openrouter:z-ai/glm-5.2", "base_url": "https://allowed.example/v1"},
        {"provider": "openrouter", "model": "allowed", "base_url": "https://open.api.z.ai/v1"},
    ):
        with pytest.raises(RoutingPolicyError):
            check_route(policy, **route)


def test_runtime_resolver_rejects_requested_route_before_discovery(monkeypatch):
    from hermes_cli import config
    import hermes_cli.runtime_provider as runtime_provider
    from hermes_cli.routing_policy import RoutingPolicyError

    monkeypatch.setattr(runtime_provider, "load_config", lambda: {
        "model": {"provider": "", "default": ""},
    })
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {
        "routing_policy": {"enabled": True, "require_explicit": True},
    })
    monkeypatch.setattr(
        runtime_provider,
        "resolve_provider",
        lambda *args, **kwargs: pytest.fail("credential discovery must not run"),
    )

    with pytest.raises(RoutingPolicyError, match="explicit provider"):
        runtime_provider.resolve_runtime_provider()


def test_runtime_resolver_rejects_final_resolved_route(monkeypatch):
    from hermes_cli import config
    import hermes_cli.runtime_provider as runtime_provider
    from hermes_cli.routing_policy import RoutingPolicyError

    monkeypatch.setattr(runtime_provider, "load_config", lambda: {
        "model": {"provider": "openrouter", "default": "z-ai/glm-5.2"},
    })
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {
        "routing_policy": {"enabled": True, "deny": {"models": ["z-ai/*"]}},
    })
    monkeypatch.setattr(runtime_provider, "_ladder_rungs", lambda *args: iter(({
        "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "test",
    },)))

    with pytest.raises(RoutingPolicyError, match="selected model"):
        runtime_provider.resolve_runtime_provider()


def test_current_policy_follows_a_to_b_to_a_profile_scopes(tmp_path, monkeypatch):
    """Scope binding, rather than a database path, selects the policy owner."""
    from agent import secret_scope
    from gateway.run import _profile_runtime_scope
    from hermes_cli.routing_policy import RoutingPolicyError, check_route, current_routing_policy

    root = tmp_path / "hermes"
    beta = root / "profiles" / "beta"
    beta.mkdir(parents=True)
    (root / "config.yaml").write_text("routing_policy:\n  enabled: true\n", encoding="utf-8")
    (beta / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: ['z-ai/*']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)

    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        with _profile_runtime_scope(root, {}):
            check_route(current_routing_policy(), provider="openrouter", model="z-ai/glm-5.2", base_url="")
        with _profile_runtime_scope(beta, {}):
            with pytest.raises(RoutingPolicyError, match="selected model"):
                check_route(current_routing_policy(), provider="openrouter", model="z-ai/glm-5.2", base_url="")
        with _profile_runtime_scope(root, {}):
            check_route(current_routing_policy(), provider="openrouter", model="z-ai/glm-5.2", base_url="")
    finally:
        secret_scope.set_multiplex_active(previous)
