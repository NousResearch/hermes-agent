"""A served profile revalidates plugin sources before its adapters are rebuilt (#144336).

``hermes -p <profile> gateway restart`` unserve+hot-serves a named profile, and the reconnect path
rebuilds a platform that was down from the same scoped registration. When the operator dropped a user
platform-plugin override into the profile's ``plugins/`` after the profile's first discovery sweep, both
paths used to rebuild from the bundled registration: ``discover_plugins()`` returned at its
``_discovered`` guard. These drive the real ``_load_secondary_profile_config`` and
``_secondary_reconnect_attempt`` and pin that the override is what the manager holds when the rebuild
reads it.
"""
import types

import pytest
import yaml

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli.plugins import _reset_plugin_managers_for_tests, get_plugin_manager

KEY = "platforms/demo"


def _write_platform_plugin(base):
    plugin_dir = base / "platforms" / "demo"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({
        "name": "demo-platform",
        "label": "Demo",
        "kind": "platform",
        "version": "0.1.0",
        "description": "hot-serve rescan probe",
    }))
    (plugin_dir / "__init__.py").write_text("def register(ctx):\n    pass\n")
    return plugin_dir


def _profile_with_one_platform(tmp_path, monkeypatch):
    """A fresh HERMES_HOME whose profile ``work`` is configured for ``KEY``, plus an empty bundled dir."""
    home = tmp_path / "hermes"
    profile_home = home / "profiles" / "work"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [KEY]}}), encoding="utf-8")
    (profile_home / ".env").write_text("", encoding="utf-8")
    bundled = tmp_path / "bundled"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    _reset_plugin_managers_for_tests()
    _write_platform_plugin(bundled)
    return profile_home


def _first_sweep(profile_home):
    """What the gateway did for this profile at start (and on the ``plugins enable`` live reload)."""
    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import discover_plugins

    with _profile_runtime_scope(profile_home, hydrate_secrets=False):
        discover_plugins()
        return get_plugin_manager()._plugins[KEY]


def _serve_runner():
    """Only the collaborators ``_load_secondary_profile_config`` itself uses."""
    runner = object.__new__(GatewayRunner)
    runner._subscribe_plugin_rewire = lambda *args, **kwargs: None
    runner._register_config_hooks = lambda *args, **kwargs: None
    runner._snapshot_profile_busy_modes = lambda *args, **kwargs: None
    return runner


def _reconnect_runner(monkeypatch, profile_home, captured):
    """Only the collaborators ``_secondary_reconnect_attempt`` reaches before it rebuilds an adapter.

    The platform being reconnected is arbitrary: the rescan covers the whole profile scope, so the
    override under ``KEY`` is revalidated for any platform's rebuild.
    """
    runner = object.__new__(GatewayRunner)
    adapter = object()

    def _create_adapter(platform, config):
        # Runs inside the profile scope, after the rescan and just before the adapter is built: this is
        # the plugin state the rebuild's platform-registration lookup reads. The adapter itself is
        # stubbed at that boundary, so this pins the registration inputs, not the built adapter.
        captured["plugin"] = get_plugin_manager()._plugins[KEY]
        return adapter

    runner._create_adapter = _create_adapter
    runner._configure_profile_adapter = lambda *args, **kwargs: None

    async def _connect(adapter_arg, platform, *, is_reconnect=False, initial=False):
        return True

    runner._connect_adapter_with_timeout = _connect

    monkeypatch.setattr("gateway.run._platform_has_bot_credential", lambda *args, **kwargs: True)
    monkeypatch.setattr("gateway.run_adapters.carry_inbound_dedup", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_home)
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda *args, **kwargs: None)
    config = types.SimpleNamespace(platforms={Platform.TELEGRAM: types.SimpleNamespace(enabled=True)})
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    return runner, Platform.TELEGRAM


@pytest.mark.asyncio
async def test_hot_serve_applies_plugin_override_added_since_the_last_sweep(tmp_path, monkeypatch):
    profile_home = _profile_with_one_platform(tmp_path, monkeypatch)
    assert _first_sweep(profile_home).manifest.source == "bundled"

    _write_platform_plugin(profile_home / "plugins")  # the operator drops the override

    from gateway.run import _profile_runtime_scope

    try:
        await _serve_runner()._load_secondary_profile_config("work", profile_home)
        with _profile_runtime_scope(profile_home, hydrate_secrets=False):
            assert get_plugin_manager()._plugins[KEY].manifest.source == "user"
    finally:
        _reset_plugin_managers_for_tests()


@pytest.mark.asyncio
async def test_hot_serve_does_not_reload_when_no_plugin_source_changed(tmp_path, monkeypatch):
    profile_home = _profile_with_one_platform(tmp_path, monkeypatch)
    before = _first_sweep(profile_home)

    from gateway.run import _profile_runtime_scope

    try:
        await _serve_runner()._load_secondary_profile_config("work", profile_home)
        with _profile_runtime_scope(profile_home, hydrate_secrets=False):
            # Same loaded record: the serve path did not unload and re-import the profile's plugins.
            assert get_plugin_manager()._plugins[KEY] is before
    finally:
        _reset_plugin_managers_for_tests()


@pytest.mark.asyncio
async def test_reconnect_rescan_sees_plugin_override_added_while_the_platform_was_down(tmp_path, monkeypatch):
    profile_home = _profile_with_one_platform(tmp_path, monkeypatch)
    assert _first_sweep(profile_home).manifest.source == "bundled"

    _write_platform_plugin(profile_home / "plugins")  # the operator drops it while the platform is down

    captured = {}
    runner, platform = _reconnect_runner(monkeypatch, profile_home, captured)
    try:
        adapter, success = await runner._secondary_reconnect_attempt("work", platform)
        assert adapter is not None
        assert success is True
        assert captured["plugin"].manifest.source == "user"
    finally:
        _reset_plugin_managers_for_tests()


@pytest.mark.asyncio
async def test_reconnect_rebuild_keeps_the_live_plugin_when_no_source_changed(tmp_path, monkeypatch):
    profile_home = _profile_with_one_platform(tmp_path, monkeypatch)
    before = _first_sweep(profile_home)

    captured = {}
    runner, platform = _reconnect_runner(monkeypatch, profile_home, captured)
    try:
        adapter, success = await runner._secondary_reconnect_attempt("work", platform)
        assert adapter is not None
        assert success is True
        # Same loaded record: a reconnect that changed no plugin input does not unload and re-import.
        assert captured["plugin"] is before
    finally:
        _reset_plugin_managers_for_tests()
