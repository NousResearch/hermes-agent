"""Tests for the model-providers plugin discovery system.

Verifies that:
 1. All bundled providers at plugins/model-providers/<name>/ are discovered
 2. User plugins at $HERMES_HOME/plugins/model-providers/<name>/ override bundled
 3. plugin.yaml manifests with kind=model-provider are correctly categorized
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock



REPO_ROOT = Path(__file__).resolve().parents[2]


def _clear_provider_caches():
    """Force providers/__init__.py to re-discover on next list_providers()."""
    import providers as _pkg
    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
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


def test_pip_entry_point_registers_provider(monkeypatch):
    """A dedicated pip entry point registers without general-plugin config."""
    import providers as provider_module

    def register() -> None:
        provider_module.register_provider(
            provider_module.ProviderProfile(
                name="packaged-provider",
                aliases=("packaged-alias",),
                base_url="https://packaged.invalid/v1",
            )
        )

    entry_point = MagicMock()
    entry_point.name = "packaged-provider"
    entry_point.load.return_value = register
    entry_points = MagicMock()
    entry_points.select.return_value = [entry_point]
    monkeypatch.setattr(
        provider_module.importlib_metadata,
        "entry_points",
        lambda: entry_points,
    )

    _clear_provider_caches()
    profile = provider_module.get_provider_profile("packaged-alias")

    assert profile is not None
    assert profile.name == "packaged-provider"
    entry_points.select.assert_called_once_with(
        group="hermes_agent.model_providers"
    )
    entry_point.load.assert_called_once_with()
    _clear_provider_caches()


def test_broken_pip_entry_point_does_not_block_others(monkeypatch, caplog):
    """One package failure is isolated from remaining provider packages."""
    import logging

    import providers as provider_module

    broken = MagicMock()
    broken.name = "broken-provider"
    broken.load.side_effect = RuntimeError("broken package")

    def register_working() -> None:
        provider_module.register_provider(
            provider_module.ProviderProfile(name="working-packaged-provider")
        )

    working = MagicMock()
    working.name = "working-provider"
    working.load.return_value = register_working
    entry_points = MagicMock()
    entry_points.select.return_value = [broken, working]
    monkeypatch.setattr(
        provider_module.importlib_metadata,
        "entry_points",
        lambda: entry_points,
    )

    _clear_provider_caches()
    with caplog.at_level(logging.WARNING, logger="providers"):
        profile = provider_module.get_provider_profile("working-packaged-provider")

    assert profile is not None
    assert any(
        "broken-provider" in record.message and "broken package" in record.message
        for record in caplog.records
    )
    _clear_provider_caches()


def test_user_directory_overrides_pip_entry_point(tmp_path, monkeypatch):
    """The profile-local directory remains more specific than site packages."""
    import providers as provider_module

    def register_packaged() -> None:
        provider_module.register_provider(
            provider_module.ProviderProfile(
                name="precedence-provider",
                base_url="https://packaged.invalid/v1",
            )
        )

    entry_point = MagicMock()
    entry_point.name = "precedence-provider"
    entry_point.load.return_value = register_packaged
    entry_points = MagicMock()
    entry_points.select.return_value = [entry_point]
    monkeypatch.setattr(
        provider_module.importlib_metadata,
        "entry_points",
        lambda: entry_points,
    )

    hermes_home = tmp_path / "hermes-home"
    user_plugin = (
        hermes_home / "plugins" / "model-providers" / "precedence-provider"
    )
    user_plugin.mkdir(parents=True)
    (user_plugin / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(\n"
        "    name='precedence-provider',\n"
        "    base_url='https://user.invalid/v1',\n"
        "))\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _clear_provider_caches()
    profile = provider_module.get_provider_profile("precedence-provider")

    assert profile is not None
    assert profile.base_url == "https://user.invalid/v1"
    _clear_provider_caches()


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
