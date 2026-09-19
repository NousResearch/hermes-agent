import pytest


def test_disabled_policy_allows_every_route():
    from hermes_cli.routing_policy import check_route

    check_route({"enabled": False}, provider="openrouter", model="z-ai/glm-5.2", base_url="https://api.z.ai/v1")


def test_enabled_policy_requires_explicit_provider_before_discovery():
    from hermes_cli.routing_policy import RoutingPolicyError, check_requested_route

    with pytest.raises(RoutingPolicyError, match="explicit provider"):
        check_requested_route({"enabled": True, "require_explicit": True}, requested_provider="auto", model="")


def test_enabled_policy_denies_glm_alias_model_and_host():
    from hermes_cli.routing_policy import RoutingPolicyError, check_route

    policy = {
        "enabled": True,
        "deny": {
            "providers": ["glm"],
            "models": ["glm-*", "z-ai/*"],
            "base_url_hosts": ["api.z.ai"],
        },
    }
    for kwargs in (
        {"provider": "zai", "model": "allowed", "base_url": "https://allowed.example/v1"},
        {"provider": "openrouter", "model": "GLM-5.2:free", "base_url": "https://allowed.example/v1"},
        {"provider": "openrouter", "model": "openrouter:z-ai/glm-5.2", "base_url": "https://allowed.example/v1"},
        {"provider": "zai", "model": "zai:glm-5.2", "base_url": "https://allowed.example/v1"},
        {"provider": "openrouter", "model": "allowed", "base_url": "https://open.api.z.ai/v1"},
        {"provider": "openrouter", "model": "allowed", "base_url": "api.z.ai/v1"},
    ):
        with pytest.raises(RoutingPolicyError):
            check_route(policy, **kwargs)


def test_model_deny_matches_prefixed_model_tail():
    from hermes_cli.routing_policy import RoutingPolicyError, check_route

    with pytest.raises(RoutingPolicyError, match="selected model"):
        check_route(
            {"enabled": True, "deny": {"models": ["glm-*"]}},
            provider="openrouter",
            model="openrouter:z-ai/glm-5.2",
            base_url="https://allowed.example/v1",
        )


def test_default_config_exposes_disabled_routing_policy():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["routing_policy"] == {"enabled": False, "require_explicit": False, "deny": {"providers": [], "models": [], "base_url_hosts": []}}


def test_config_path_owner_uses_installation_root_while_another_profile_is_active(tmp_path, monkeypatch):
    """A config path is owned by its real profile, not by active HERMES_HOME."""
    from hermes_cli.routing_policy import profile_home_for_config_path

    root = tmp_path / "hermes"
    default = root / "config.yaml"
    alpha = root / "profiles" / "alpha" / "config.yaml"
    beta = root / "profiles" / "beta" / "config.yaml"
    for config in (default, alpha, beta):
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("routing_policy: {}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(beta.parent))

    assert profile_home_for_config_path(alpha) == alpha.parent
    assert profile_home_for_config_path(default) == root


def test_config_path_owner_rejects_invalid_profile_lookalike(tmp_path, monkeypatch):
    """A directory under profiles is not trusted unless it has a valid profile id."""
    from hermes_cli.routing_policy import profile_home_for_config_path

    root = tmp_path / "hermes"
    active = root / "profiles" / "beta"
    lookalike = root / "profiles" / "Not-A-Profile" / "config.yaml"
    for config in (active / "config.yaml", lookalike):
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("routing_policy: {}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(active))

    assert profile_home_for_config_path(lookalike) is None


def test_config_path_owner_preserves_logical_named_profile_symlink(tmp_path, monkeypatch):
    """A config reached through ``profiles/<name>`` retains that trusted owner."""
    from hermes_cli.routing_policy import profile_home_for_config_path

    root = tmp_path / "hermes"
    logical_profile = root / "profiles" / "restricted"
    outside_profile = tmp_path / "outside" / "restricted"
    root.mkdir()
    logical_profile.parent.mkdir()
    outside_profile.mkdir(parents=True)
    logical_profile.symlink_to(outside_profile, target_is_directory=True)
    (root / "config.yaml").write_text("routing_policy: {}\n", encoding="utf-8")
    (outside_profile / "config.yaml").write_text("routing_policy: {}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    assert profile_home_for_config_path(logical_profile / "config.yaml") == logical_profile


def test_runtime_policy_rejects_auto_before_credential_discovery(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider
    from hermes_cli.routing_policy import RoutingPolicyError

    monkeypatch.setattr(runtime_provider, "load_config", lambda: {
        "model": {"provider": "", "default": ""},
        "routing_policy": {"enabled": True, "require_explicit": True},
    })
    monkeypatch.setattr(
        runtime_provider,
        "resolve_provider",
        lambda *args, **kwargs: pytest.fail("credential discovery must not run"),
    )

    with pytest.raises(RoutingPolicyError, match="explicit provider"):
        runtime_provider.resolve_runtime_provider()


def test_runtime_policy_denies_resolved_route_before_return(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider
    from hermes_cli.routing_policy import RoutingPolicyError

    monkeypatch.setattr(runtime_provider, "load_config", lambda: {
        "model": {"provider": "openrouter", "default": "z-ai/glm-5.2"},
        "routing_policy": {"enabled": True, "deny": {"models": ["z-ai/*"]}},
    })
    monkeypatch.setattr(runtime_provider, "_ladder_rungs", lambda *args: iter(({
        "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "test",
    },)))

    with pytest.raises(RoutingPolicyError, match="selected model"):
        runtime_provider.resolve_runtime_provider()


def test_policy_keeps_unrelated_route_usable():
    from hermes_cli.routing_policy import check_route

    check_route(
        {"enabled": True, "deny": {"providers": ["glm"], "models": ["glm-*"], "base_url_hosts": ["api.z.ai"]}},
        provider="openai-codex",
        model="gpt-5.6-terra",
        base_url="https://chatgpt.com/backend-api/codex",
    )
