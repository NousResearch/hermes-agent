"""Built-in auth pools must not be shadowed by compatible provider config."""

import json
from types import SimpleNamespace

import pytest
import yaml


@pytest.fixture
def isolated_auth_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    # Pool loading must not discover real Claude Code/Keychain credentials.
    monkeypatch.setattr("agent.anthropic_adapter.read_claude_code_credentials", lambda: None)
    monkeypatch.setattr("agent.anthropic_adapter.read_hermes_oauth_credentials", lambda: None)
    return tmp_path


@pytest.mark.parametrize("provider", ["anthropic", "deepseek", "openrouter"])
@pytest.mark.parametrize("target", ["1", "e6a714"])
def test_remove_builtin_with_provider_config(isolated_auth_home, capsys, provider, target):
    from hermes_cli.auth_commands import auth_remove_command

    home = isolated_auth_home
    (home / "config.yaml").write_text(yaml.safe_dump({
        "providers": {provider: {"api": "https://provider.example/v1"}},
    }))
    entries = [
        {
            "id": entry_id,
            "label": label,
            "auth_type": "oauth" if provider == "anthropic" else "api_key",
            "source": "manual",
            "access_token": f"fake-token-{priority}",
            "priority": priority,
        }
        for priority, (entry_id, label) in enumerate([
            ("e6a714", "dashboard PKCE"), ("f7b825", "hermes_pkce"),
        ])
    ]
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {provider: entries},
    }))

    auth_remove_command(SimpleNamespace(provider=provider, target=target))

    payload = json.loads((home / "auth.json").read_text())
    assert [entry["id"] for entry in payload["credential_pool"][provider]] == ["f7b825"]
    assert payload["credential_pool"][provider][0]["priority"] == 0
    assert f"custom:{provider}" not in payload["credential_pool"]
    assert capsys.readouterr().out == f"Removed {provider} credential #1 (dashboard PKCE)\n"


@pytest.mark.parametrize("schema", ["providers", "custom_providers"])
def test_builtin_ids_win_without_breaking_custom_names(isolated_auth_home, schema):
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.auth_commands import _normalize_provider

    builtins = sorted({*PROVIDER_REGISTRY, "openrouter"})
    names = [*builtins, "My Endpoint"]
    if schema == "providers":
        config = {schema: {
            name: {"api": "https://provider.example/v1"} for name in names
        }}
    else:
        config = {schema: [
            {"name": name, "base_url": "https://provider.example/v1"} for name in names
        ]}
    (isolated_auth_home / "config.yaml").write_text(yaml.safe_dump(config))

    for provider in builtins:
        assert _normalize_provider(f" {provider.upper()} ") == provider
        assert _normalize_provider(f"custom:{provider}") == f"custom:{provider}"
    assert _normalize_provider("My Endpoint") == "custom:my-endpoint"
    assert _normalize_provider("or") == "openrouter"
    assert _normalize_provider("open-router") == "openrouter"
    assert _normalize_provider("grok-oauth") == "xai-oauth"
    assert _normalize_provider("unknown-provider") == "unknown-provider"
