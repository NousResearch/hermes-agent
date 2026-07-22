"""Tests for the model-providers plugin discovery system.

Verifies that:
 1. All bundled providers at plugins/model-providers/<name>/ are discovered
 2. User plugins at $HERMES_HOME/plugins/model-providers/<name>/ override bundled
 3. plugin.yaml manifests with kind=model-provider are correctly categorized
"""

from __future__ import annotations

import sys
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[2]


def _clear_provider_caches():
    """Force providers/__init__.py to re-discover on next list_providers()."""
    import providers as _pkg
    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._SOURCES.clear()
    _pkg._OVERRIDES.clear()
    _pkg._discovered = False
    # Evict any cached plugin modules so the next import re-executes.
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("plugins.model_providers")
            or mod.startswith("_hermes_user_provider")
        ):
            del sys.modules[mod]


def test_bundled_plugins_discovered():
    """Every plugins/model-providers/<name>/ should contain a plugin.yaml + __init__.py."""
    plugins_dir = REPO_ROOT / "plugins" / "model-providers"
    assert plugins_dir.is_dir(), f"Missing {plugins_dir}"

    child_dirs = [c for c in plugins_dir.iterdir() if c.is_dir()]
    assert len(child_dirs) >= 28, f"Expected at least 28 provider plugins, found {len(child_dirs)}"

    for child in child_dirs:
        assert (child / "__init__.py").exists(), f"{child.name} missing __init__.py"
        assert (child / "plugin.yaml").exists(), f"{child.name} missing plugin.yaml"


def test_all_profiles_register():
    """After discovery, the registry must contain every bundled provider directory.

    This is an invariant — the number of profiles matches the number of plugin
    directories, not a hardcoded count. Counts shift when providers are
    added/removed; that's expected and shouldn't break CI.
    """
    _clear_provider_caches()
    from providers import list_providers

    plugins_dir = REPO_ROOT / "plugins" / "model-providers"
    plugin_dir_count = sum(1 for c in plugins_dir.iterdir() if c.is_dir())

    profiles = list_providers()
    names = sorted(p.name for p in profiles)
    # Some plugin __init__.py files register multiple profiles, so the registry
    # count is >= the directory count (never less).
    assert len(names) >= plugin_dir_count, (
        f"Expected at least {plugin_dir_count} profiles (one per plugin dir), got {len(names)}: {names}"
    )

    # Spot-check representative providers from different categories
    for required in (
        "openrouter", "anthropic", "custom", "bedrock", "openai-codex",
        "minimax-oauth", "gmi", "xiaomi", "alibaba-coding-plan", "fireworks",
    ):
        assert required in names, f"Missing profile: {required}"


def test_user_plugin_overrides_bundled(tmp_path, monkeypatch):
    """A user plugin with the same name must override the bundled profile."""
    # Point HERMES_HOME at a fresh temp dir
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # get_hermes_home() may be module-cached depending on codebase; ensure the
    # env var is the source of truth. Most code paths re-read it each call.

    # Drop a user plugin that replaces 'gmi'
    user_gmi = hermes_home / "plugins" / "model-providers" / "gmi"
    user_gmi.mkdir(parents=True)
    (user_gmi / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "\n"
        "custom_gmi = ProviderProfile(\n"
        '    name="gmi",\n'
        '    aliases=("gmi-user-override-test",),\n'
        '    env_vars=("GMI_API_KEY",),\n'
        '    base_url="https://user-override.example.com/v1",\n'
        '    auth_type="api_key",\n'
        ")\n"
        "register_provider(custom_gmi)\n"
    )
    (user_gmi / "plugin.yaml").write_text(
        "name: gmi-user-override\n"
        "kind: model-provider\n"
        "version: 0.0.1\n"
        "description: Test user override\n"
    )

    _clear_provider_caches()
    from providers import get_provider_profile

    gmi = get_provider_profile("gmi")
    assert gmi is not None
    assert gmi.base_url == "https://user-override.example.com/v1", (
        f"User override not applied; got base_url={gmi.base_url!r}"
    )
    assert "gmi-user-override-test" in gmi.aliases

    # Clean up: reset discovery state so other tests see the bundled version
    _clear_provider_caches()


def test_bundled_profiles_report_source_bundled():
    """Every bundled profile should be tagged with source='bundled'."""
    _clear_provider_caches()
    from providers import get_provider_source, list_providers

    profiles = list_providers()
    assert profiles, "Discovery returned no profiles"
    # Sample across several built-ins; if any of these are missing the change
    # has bigger problems than source tracking.
    for required in ("gmi", "openrouter", "anthropic", "deepseek"):
        assert get_provider_source(required) == "bundled", (
            f"{required} expected source='bundled', got {get_provider_source(required)!r}"
        )
    # Aliases must resolve to the canonical profile's source.
    from providers import get_provider_profile

    for p in profiles:
        for alias in p.aliases:
            assert get_provider_source(alias) == get_provider_source(p.name), (
                f"alias {alias!r} of {p.name!r} reports different source"
            )
            assert get_provider_profile(alias) is p


def test_user_override_records_displaced_source(tmp_path, monkeypatch):
    """A user plugin that overrides a bundled profile must update source +
    record the displaced 'bundled' source in list_provider_overrides()."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    user_gmi = hermes_home / "plugins" / "model-providers" / "gmi"
    user_gmi.mkdir(parents=True)
    (user_gmi / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(\n"
        '    name="gmi",\n'
        '    aliases=(),\n'
        '    env_vars=("GMI_API_KEY",),\n'
        '    base_url="https://user-override.example/v1",\n'
        '    auth_type="api_key",\n'
        "))\n"
    )
    (user_gmi / "plugin.yaml").write_text(
        "name: gmi\nkind: model-provider\nversion: 0.0.1\n"
    )

    _clear_provider_caches()
    from providers import (
        get_provider_profile,
        get_provider_source,
        list_provider_overrides,
    )

    assert get_provider_source("gmi") == "user"
    overrides = list_provider_overrides()
    assert "gmi" in overrides, f"gmi missing from overrides: {overrides}"
    assert overrides["gmi"] == ["bundled"]
    # Returned mapping must be a defensive copy.
    overrides["gmi"].append("tampered")
    fresh = list_provider_overrides()
    assert fresh["gmi"] == ["bundled"]
    # Active profile is the user one.
    profile = get_provider_profile("gmi")
    assert profile is not None
    assert profile.base_url == "https://user-override.example/v1"

    _clear_provider_caches()


def test_get_provider_source_unknown_returns_none():
    """Unregistered names return None rather than raising."""
    _clear_provider_caches()
    from providers import get_provider_source

    assert get_provider_source("definitely-not-a-real-provider-xyz") is None


def test_general_plugin_manager_skips_model_provider_kind(tmp_path, monkeypatch):
    """The general PluginManager must NOT import model-provider plugins
    (providers/__init__.py handles them). It records the manifest only."""
    from hermes_cli import plugins as plugin_mod

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Create a user-installed plugin with an explicit kind: model-provider.
    user_plugin = hermes_home / "plugins" / "test-model-provider"
    user_plugin.mkdir(parents=True)
    (user_plugin / "plugin.yaml").write_text(
        "name: test-model-provider\n"
        "kind: model-provider\n"
        "version: 0.0.1\n"
    )
    (user_plugin / "__init__.py").write_text(
        # Intentionally broken import — if the general loader tries to
        # import this module, the test will fail with ImportError.
        "raise AssertionError('model-provider plugins must not be imported by PluginManager')\n"
    )

    # Fresh manager
    manager = plugin_mod.PluginManager()
    manager.discover_and_load(force=True)

    # The manifest should be recorded but not loaded
    loaded = manager._plugins.get("test-model-provider")
    assert loaded is not None
    assert loaded.manifest.kind == "model-provider"
    # No import means the module must NOT be in the plugins list as a loaded one.
    # We check that the general loader didn't crash and didn't raise from the
    # broken __init__.py.
