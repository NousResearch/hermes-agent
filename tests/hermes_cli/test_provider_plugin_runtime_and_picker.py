"""Regression tests for model-provider plugins in runtime resolution and /model."""

from __future__ import annotations

import sys

from hermes_cli.auth import (
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.runtime_provider import resolve_runtime_provider


def _clear_provider_caches():
    import providers as _pkg

    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._discovered = False
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("plugins.model_providers")
            or mod.startswith("_hermes_user_provider")
        ):
            del sys.modules[mod]


def _install_test_provider(tmp_path, *, api_mode="chat_completions"):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(exist_ok=True)
    plugin_dir = hermes_home / "plugins" / "model-providers" / "relaybox"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "\n"
        "register_provider(ProviderProfile(\n"
        '    name="relaybox",\n'
        '    aliases=("rbx",),\n'
        '    display_name="RelayBox",\n'
        '    description="RelayBox test provider",\n'
        '    env_vars=("RELAYBOX_API_KEY",),\n'
        '    base_url="https://relaybox.example/v1",\n'
        '    auth_type="api_key",\n'
        f'    api_mode="{api_mode}",\n'
        '    fallback_models=("relaybox-agent", "relaybox-fast"),\n'
        "))\n"
    )
    (plugin_dir / "plugin.yaml").write_text(
        "name: relaybox\n"
        "kind: model-provider\n"
        "version: 0.0.1\n"
        "description: Test runtime + picker provider\n"
    )
    return hermes_home


def test_plugin_provider_resolves_credentials_and_status(tmp_path, monkeypatch):
    hermes_home = _install_test_provider(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("RELAYBOX_API_KEY", "relaybox-secret")
    _clear_provider_caches()

    creds = resolve_api_key_provider_credentials("relaybox")
    status = get_api_key_provider_status("relaybox")

    assert creds["provider"] == "relaybox"
    assert creds["api_key"] == "relaybox-secret"
    assert creds["base_url"] == "https://relaybox.example/v1"
    assert creds["source"] == "RELAYBOX_API_KEY"
    assert status["configured"] is True
    assert status["logged_in"] is True
    assert status["base_url"] == "https://relaybox.example/v1"
    assert status["key_source"] == "RELAYBOX_API_KEY"

    _clear_provider_caches()


def test_plugin_provider_alias_and_runtime_resolution(tmp_path, monkeypatch):
    hermes_home = _install_test_provider(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("RELAYBOX_API_KEY", "relaybox-secret")
    _clear_provider_caches()

    assert resolve_provider("rbx") == "relaybox"

    runtime = resolve_runtime_provider(requested="rbx")

    assert runtime["provider"] == "relaybox"
    assert runtime["requested_provider"] == "rbx"
    assert runtime["api_key"] == "relaybox-secret"
    assert runtime["base_url"] == "https://relaybox.example/v1"
    assert runtime["api_mode"] == "chat_completions"

    _clear_provider_caches()


def test_plugin_provider_appears_in_model_picker(tmp_path, monkeypatch):
    hermes_home = _install_test_provider(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("RELAYBOX_API_KEY", "relaybox-secret")
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    _clear_provider_caches()

    providers = list_authenticated_providers(
        current_provider="openrouter",
        user_providers={},
        custom_providers=[],
        max_models=10,
    )

    relaybox = next((p for p in providers if p["slug"] == "relaybox"), None)
    assert relaybox is not None
    assert relaybox["name"] == "RelayBox"
    assert relaybox["models"] == ["relaybox-agent", "relaybox-fast"]
    assert relaybox["total_models"] == 2
    assert relaybox["source"] == "plugin"

    _clear_provider_caches()


def test_plugin_provider_custom_api_mode_in_runtime(tmp_path, monkeypatch):
    hermes_home = _install_test_provider(tmp_path, api_mode="anthropic_messages")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("RELAYBOX_API_KEY", "relaybox-secret")
    _clear_provider_caches()

    runtime = resolve_runtime_provider(requested="relaybox")

    assert runtime["provider"] == "relaybox"
    assert runtime["api_mode"] == "anthropic_messages"

    _clear_provider_caches()


def test_plugin_provider_picker_reads_dotenv(tmp_path, monkeypatch):
    hermes_home = _install_test_provider(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("RELAYBOX_API_KEY", raising=False)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda key: "dotenv-secret" if key == "RELAYBOX_API_KEY" else None,
    )
    _clear_provider_caches()

    providers = list_authenticated_providers(
        current_provider="openrouter",
        user_providers={},
        custom_providers=[],
        max_models=10,
    )

    relaybox = next((p for p in providers if p["slug"] == "relaybox"), None)
    assert relaybox is not None, "Plugin provider with .env-only key should appear in picker"
    assert relaybox["models"] == ["relaybox-agent", "relaybox-fast"]

    _clear_provider_caches()
