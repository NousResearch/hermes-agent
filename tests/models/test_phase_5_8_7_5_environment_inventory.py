"""Phase 5.8.7.5 hard-cut gates for scoped endpoints and read-only inventory."""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

from application_configured_provider_facts import custom_identity, requested_provider
from hermes_cli.inventory import ConfigContext, build_models_payload
from providers.environment import declared_endpoint_override

ROOT = Path(__file__).resolve().parents[2]


def _imports(rel: str) -> set[tuple[str, str]]:
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    return {
        (node.module, item.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for item in node.names
    }


def test_endpoint_declaration_never_uses_credential_env_vars():
    assert declared_endpoint_override(
        base_url_env_var="RAMP_ROUTER_BASE_URL",
        environment={"RAMP_ROUTER_API_KEY": "secret"},
    ) == ""
    assert declared_endpoint_override(
        base_url_env_var="RAMP_ROUTER_BASE_URL",
        environment={"RAMP_ROUTER_BASE_URL": "https://example.test/v1/"},
    ) == "https://example.test/v1"
    assert declared_endpoint_override(
        base_url_env_var="RAMP_ROUTER_BASE_URL",
        configured="https://configured.test/v1/",
        environment={"RAMP_ROUTER_BASE_URL": "https://environment.test"},
    ) == "https://configured.test/v1"


def test_requested_provider_config_takes_precedence_over_environment():
    assert requested_provider(
        {"model": {"provider": " AnThRoPiC "}},
        environment="openrouter",
    ) == "anthropic"
    assert requested_provider({"model": {}}, environment="opencode-go") == "opencode-go"
    assert requested_provider({"model": {}}, environment="") == "auto"


def test_configured_identity_uses_supplied_profile_config_without_cli_runtime():
    config = {
        "model": {"provider": "custom"},
        "providers": {"lab": {"name": "Laboratory", "base_url": "https://lab.test/v1"}},
        "custom_providers": [],
    }
    assert custom_identity(
        base_url="https://lab.test/v1", config=config,
    ) == "custom:lab"
    assert custom_identity(
        base_url="https://unknown.test/v1", config=config,
    ) == ""


def test_custom_identity_can_use_a_scoped_environment_candidate(monkeypatch):
    import agent.secret_scope as secret_scope

    monkeypatch.setattr(
        secret_scope, "get_secret",
        lambda name, default="": "lab" if name == "HERMES_INFERENCE_PROVIDER" else default,
    )
    cfg = {
        "model": {},
        "providers": {"lab": {"name": "Laboratory", "base_url": "https://lab.test/v1"}},
    }
    assert custom_identity(config=cfg) == "custom:lab"


def test_managed_endpoint_identity_requires_a_proven_endpoint(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.local_runtime.endpoint._state_endpoint",
        lambda: {"base_url": "http://127.0.0.1:8080/v1"},
    )
    cfg = {"model": {"provider": "custom"}, "providers": {}, "custom_providers": []}
    assert custom_identity(base_url="http://127.0.0.1:8080/v1", config=cfg) == "llamacpp"
    assert custom_identity(base_url="http://127.0.0.1:9000/v1", config=cfg) == ""


def test_inventory_observation_cannot_persist_custom_catalogue():
    ctx = ConfigContext(
        current_provider="custom:lab",
        current_model="sample",
        current_base_url="https://lab.test/v1",
        user_providers={},
        custom_providers=[{"name": "lab", "base_url": "https://lab.test/v1"}],
    )
    rows = [{
        "slug": "custom:lab", "name": "Lab", "models": ["sample", "new"],
        "total_models": 2, "is_current": True,
        "is_user_defined": True, "source": "user-config",
    }]
    with (
        patch("application_provider_discovery.list_authenticated_providers", return_value=rows),
        patch("hermes_cli.config.save_config", side_effect=AssertionError("GET saved config")) as saved,
        patch("hermes_cli.inventory._local_runtime_row", return_value=None),
        patch("hermes_cli.inventory._moa_provider_row", return_value=None),
    ):
        payload = build_models_payload(ctx)
    assert payload["providers"][0]["models"] == ["sample", "new"]
    saved.assert_not_called()


def test_owner_has_no_read_time_save_and_picker_imports_new_owner():
    assert not (ROOT / "hermes_cli/model_switch_providers.py").exists()
    assert ("application_provider_discovery", "list_authenticated_providers") in (
        _imports("hermes_cli/inventory.py")
    )
    tree = ast.parse((ROOT / "application_provider_discovery.py").read_text(encoding="utf-8"))
    names = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert "_save_discovered_models_to_config" not in names
    assert "save_config" not in names
    assert ("application_discovered_catalog_persistence", "_save_discovered_models_to_config") in (
        _imports("hermes_cli/model_setup_flows_custom.py")
    )


def test_tui_carryovers_no_longer_import_cli_route_decisions():
    expected = {
        "tui_gateway/methods_complete_helpers.py": "custom_identity",
        "tui_gateway/session_workdir.py": "custom_identity",
        "tui_gateway/methods_session_model_guard.py": "requested_provider",
    }
    for path, name in expected.items():
        imports = _imports(path)
        assert ("application_configured_provider_facts", name) in imports
        assert not any(
            module == "hermes_cli.runtime_provider"
            and symbol in {"canonical_custom_identity", "resolve_requested_provider"}
            for module, symbol in imports
        )


def test_scoped_endpoints_use_only_the_profile_declaration(monkeypatch):
    from types import SimpleNamespace

    import application_provider_environment as application
    import hermes_cli.config as config
    import agent.secret_scope as secret_scope

    monkeypatch.setattr(
        application, "get_provider_profile",
        lambda _name: SimpleNamespace(base_url_env_var="RAMP_ROUTER_BASE_URL"),
    )
    monkeypatch.setattr(
        config, "get_env_value_prefer_dotenv",
        lambda name: "https://profile.example/v1" if name == "RAMP_ROUTER_BASE_URL" else "",
    )
    monkeypatch.setattr(secret_scope, "current_secret_scope", lambda: None)
    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: False)
    assert application.scoped_endpoint_override(
        "router", config={"model": {"provider": "other"}}
    ) == "https://profile.example/v1"
    assert application.scoped_endpoint_override(
        "router", config={"model": {
            "provider": "router", "base_url": "https://explicit.example/v1",
        }}
    ) == "https://explicit.example/v1"


def test_deepinfra_pricing_scope_uses_canonical_endpoint(monkeypatch):
    from application_model_pricing import pricing_cache_scope
    monkeypatch.setattr(
        "application_deepinfra_catalog.deepinfra_base_url",
        lambda: "https://scoped.example/v1",
    )
    assert pricing_cache_scope("deepinfra") == "https://scoped.example/v1"


def test_router_and_usage_plugins_have_no_cli_route_bundle_dependency():
    router = _imports("plugins/model-providers/router/__init__.py")
    opencode = _imports("plugins/model-providers/opencode-zen/__init__.py")
    assert ("application_provider_environment", "scoped_endpoint_override") in router
    assert ("hermes_cli.auth", "resolve_api_key_provider_credentials") in opencode
    assert ("hermes_cli.runtime_provider", "resolve_runtime_provider") not in opencode
