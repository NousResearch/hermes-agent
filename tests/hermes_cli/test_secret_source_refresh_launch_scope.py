"""A plugin rediscovery inside a launch-profile body must not shrink that body's secret scope.

Under multiplexing the launch profile's bound scope is ``launch_secret_scope``: its ``.env`` over
the env frozen at activation, which is where systemd / ``op run`` / docker ``-e`` credentials and
the YAML-bridged allowlists (``telegram.allow_from`` -> ``TELEGRAM_ALLOWED_USERS``) live. A forced
discovery that re-applies plugin secret sources refreshes the installed scope in place; rebuilt
from the files alone it dropped those env-only values for the rest of the body.
"""
from __future__ import annotations

import os

import pytest

_FIXTURE_PLUGIN = (
    "from agent.secret_sources.base import SECRET_SOURCE_API_VERSION, FetchResult, SecretSource\n\n"
    "class FixtureVault(SecretSource):\n"
    "    name = 'fixturevault'\n"
    "    label = 'Fixture vault'\n"
    "    api_version = SECRET_SOURCE_API_VERSION\n"
    "    shape = 'bulk'\n\n"
    "    def fetch(self, cfg, home_path):\n"
    "        return FetchResult(secrets={'VAULT_ONLY_KEY': 'from-plugin'})\n\n"
    "def register(ctx):\n"
    "    ctx.register_secret_source(FixtureVault())\n"
)


@pytest.fixture
def launch_home(tmp_path, monkeypatch):
    import agent.secret_sources.registry as reg
    from hermes_cli import env_loader

    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "fixture-secret-source"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: fixture-secret-source\nversion: 0.1.0\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(_FIXTURE_PLUGIN, encoding="utf-8")
    (home / "profiles" / "ops").mkdir(parents=True)
    (home / "config.yaml").write_text(
        "allow_all_users: true\n"
        "telegram:\n  allow_from: ['111']\n"
        "plugins:\n  enabled: [fixture-secret-source]\n"
        "secrets:\n  fixturevault:\n    enabled: true\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("TELEGRAM_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "VAULT_ONLY_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SYSTEMD_ONLY_TOKEN", "env-only")  # e.g. systemd Environment=, no .env line
    reg._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()
    try:
        yield home
    finally:
        os.environ.pop("VAULT_ONLY_KEY", None)
        reg._reset_registry_for_tests()
        env_loader.reset_secret_source_cache()


def test_forced_rediscovery_keeps_launch_env_in_the_bound_launch_scope(launch_home):
    from agent.secret_scope import current_secret_scope, set_multiplex_active
    from gateway.config import load_gateway_config
    from gateway.platforms._shared import platform_gate_env
    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting

    load_gateway_config()  # the launch YAML bridge: allow_from -> TELEGRAM_ALLOWED_USERS, allow-all grant
    assert os.environ["TELEGRAM_ALLOWED_USERS"] == "111"
    manager = get_plugin_manager()
    manager.discover_and_load()  # gateway boot discovery
    activate_multi_profile_hosting()
    try:
        with _profile_runtime_scope(launch_home):  # a launch-profile turn
            assert platform_gate_env("TELEGRAM_ALLOWED_USERS") == "111"
            # A tool in the turn force-refreshes discovery (image_gen / TTS / STT "provider missing" retry).
            manager.discover_and_load(force=True)
            scope = current_secret_scope()
            assert scope.get("VAULT_ONLY_KEY") == "from-plugin"
            assert platform_gate_env("GATEWAY_ALLOW_ALL_USERS") == "true"
            # The list that narrows the grant, and the env-only credential, survive the refresh.
            assert platform_gate_env("TELEGRAM_ALLOWED_USERS") == "111"
            assert scope.get("SYSTEMD_ONLY_TOKEN") == "env-only"
        # A secondary's refresh stays files-only: the launch env never reaches its scope.
        from agent.secret_scope import refresh_installed_secret_scope
        with _profile_runtime_scope(launch_home / "profiles" / "ops"):
            assert refresh_installed_secret_scope(launch_home / "profiles" / "ops") is True
            assert "SYSTEMD_ONLY_TOKEN" not in (current_secret_scope() or {})
            assert platform_gate_env("TELEGRAM_ALLOWED_USERS") == ""
    finally:
        set_multiplex_active(False)


def test_a_refresh_for_one_home_never_replaces_another_homes_installed_scope(launch_home):
    """A scope stamped for a secondary (no home override) is not the launch home's to refresh: the
    launch profile's files and frozen env must not replace it."""
    from agent.secret_scope import (current_secret_scope, refresh_installed_secret_scope,
                                    reset_secret_scope, set_multiplex_active, set_secret_scope)
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting

    (launch_home / ".env").write_text("LAUNCH_FILE_KEY=launch\n", encoding="utf-8")
    ops = launch_home / "profiles" / "ops"
    activate_multi_profile_hosting()
    token = set_secret_scope({"OPS_KEY": "ops"}, profile_home=str(ops))
    try:
        assert refresh_installed_secret_scope(launch_home) is False
        assert dict(current_secret_scope()) == {"OPS_KEY": "ops"}
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


def test_an_unstamped_scope_on_a_pinned_host_belongs_to_the_pinned_home(tmp_path, monkeypatch):
    """A host that pins its launch home and mirrors the served profile into HERMES_HOME (Hermes
    WebUI): an unstamped scope is the pinned launch home's, so the mirrored profile's refresh must
    not replace it, and a symlinked spelling of the pinned home still owns it."""
    from agent import secret_scope
    from hermes_constants import pin_process_hermes_home

    launch = tmp_path / "launch"
    ops = launch / "profiles" / "ops"
    ops.mkdir(parents=True)
    (ops / ".env").write_text("OPS_KEY=o\n", encoding="utf-8")
    link = tmp_path / "launch-link"
    link.symlink_to(launch, target_is_directory=True)
    pin_process_hermes_home(launch)
    monkeypatch.setenv("HERMES_HOME", str(ops))  # the mirror
    token = secret_scope.set_secret_scope({"LAUNCH_KEY": "l"})
    try:
        assert secret_scope.refresh_installed_secret_scope(ops) is False
        assert dict(secret_scope.current_secret_scope()) == {"LAUNCH_KEY": "l"}
        assert secret_scope.refresh_installed_secret_scope(link) is True
        assert "OPS_KEY" not in secret_scope.current_secret_scope()
    finally:
        secret_scope.reset_secret_scope(token)
        pin_process_hermes_home(None)
