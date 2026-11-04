"""A multiplexed gateway's secondary profile must see plugin-registered secret sources (#149014).

``_load_profile_secret_scope`` hydrates a profile's secrets under its home override, but a
plugin-registered secret source (``ctx.register_secret_source()``) only exists after plugin
discovery. Without ``discover_plugins()`` before ``hydrate_profile_secret_sources``, a
secondary profile whose plugins had not loaded hydrated with an empty source registry and
``${VAR}`` placeholders in MCP server url/headers stayed unresolved. Discovery must run under
the home override so the profile's own plugins load, not the launch profile's. Mirrors the
cron-worker invariant in ``tests/cron/test_external_worker_secret_sources.py`` (#121929).
"""
from __future__ import annotations

import pytest

STUB_PLUGIN_INIT = '''
from agent.secret_sources.base import SECRET_SOURCE_API_VERSION, FetchResult, SecretSource


class TestVaultSource(SecretSource):
    api_version = SECRET_SOURCE_API_VERSION
    name = "testvault"
    label = "Test Vault"
    shape = "bulk"

    def fetch(self, cfg, home_path):
        return FetchResult(secrets={"TESTVAULT_API_KEY": "stub-vault-key"})


def register(ctx):
    ctx.register_secret_source(TestVaultSource())
'''


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A launch home with no plugins and a profile home shipping the vault plugin."""
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    profile = tmp_path / "profile"
    plugin_dir = profile / "plugins" / "test-vault"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: test-vault\ndescription: stub\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(STUB_PLUGIN_INIT, encoding="utf-8")
    (profile / "config.yaml").write_text(
        "secrets:\n  sources: [testvault]\n  testvault:\n    enabled: true\n"
        "plugins:\n  enabled: [test-vault]\n",
        encoding="utf-8",
    )
    yield launch, profile
    from agent.secret_sources import registry
    from hermes_cli.env_loader import reset_secret_source_cache
    from hermes_cli.plugins import _reset_plugin_managers_for_tests

    registry._reset_registry_for_tests()
    reset_secret_source_cache()
    _reset_plugin_managers_for_tests()


def test_load_profile_secret_scope_hydrates_owning_profile_plugin_secret_source(homes):
    """The launch home has no plugins; the profile home ships a register_secret_source plugin.
    ``_load_profile_secret_scope`` discovers plugins under the home override before hydrating,
    so the plugin's value reaches the profile's secret scope (#149014)."""
    launch, profile = homes
    from gateway.run import _load_profile_secret_scope

    assert "TESTVAULT_API_KEY" not in _load_profile_secret_scope(launch)
    assert _load_profile_secret_scope(profile).get("TESTVAULT_API_KEY") == "stub-vault-key"
