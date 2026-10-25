"""The profile hot-serve path revalidates plugin sources before rebuilding adapters (#144336).

``hermes -p <profile> gateway restart`` unserve+hot-serves a named profile. When the operator dropped a
user platform-plugin override into the profile's ``plugins/`` after the profile's first discovery
sweep, the re-serve used to rebuild the adapters from the bundled registration: ``discover_plugins()``
returned at its ``_discovered`` guard. This drives the real ``_load_secondary_profile_config`` and pins
that the override is what the manager holds afterwards.
"""
import pytest
import yaml

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


def _serve_runner():
    """Only the collaborators ``_load_secondary_profile_config`` itself uses."""
    runner = object.__new__(GatewayRunner)
    runner._subscribe_plugin_rewire = lambda *args, **kwargs: None
    runner._register_config_hooks = lambda *args, **kwargs: None
    runner._snapshot_profile_busy_modes = lambda *args, **kwargs: None
    return runner


@pytest.mark.asyncio
async def test_hot_serve_applies_plugin_override_added_since_the_last_sweep(tmp_path, monkeypatch):
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

    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import discover_plugins

    _write_platform_plugin(bundled)
    # What the gateway did for this profile at start (and on the `plugins enable` live reload).
    with _profile_runtime_scope(profile_home, hydrate_secrets=False):
        discover_plugins()
        assert get_plugin_manager()._plugins[KEY].manifest.source == "bundled"

    _write_platform_plugin(profile_home / "plugins")  # the operator drops the override

    try:
        await _serve_runner()._load_secondary_profile_config("work", profile_home)
        with _profile_runtime_scope(profile_home, hydrate_secrets=False):
            assert get_plugin_manager()._plugins[KEY].manifest.source == "user"
    finally:
        _reset_plugin_managers_for_tests()


@pytest.mark.asyncio
async def test_hot_serve_does_not_reload_when_no_plugin_source_changed(tmp_path, monkeypatch):
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

    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import discover_plugins

    _write_platform_plugin(bundled)
    with _profile_runtime_scope(profile_home, hydrate_secrets=False):
        discover_plugins()
        before = get_plugin_manager()._plugins[KEY]

    try:
        await _serve_runner()._load_secondary_profile_config("work", profile_home)
        with _profile_runtime_scope(profile_home, hydrate_secrets=False):
            # Same loaded record: the serve path did not unload and re-import the profile's plugins.
            assert get_plugin_manager()._plugins[KEY] is before
    finally:
        _reset_plugin_managers_for_tests()
