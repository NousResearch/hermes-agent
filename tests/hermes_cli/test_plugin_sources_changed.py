"""Directory plugin sources are re-checked before a profile's adapters are rebuilt (#144336).

A user platform-plugin override placed after a profile's first discovery sweep used to be invisible to
``hermes -p <profile> gateway restart``: ``discover_plugins()`` returned at its ``_discovered`` guard, so
the re-served profile rebuilt its adapters from the bundled registration. These pin the staleness probe
the hot-serve path now consults, and that a rescan happens only when the sources actually changed.
"""
import shutil

import pytest
import yaml

from hermes_cli.plugins import (
    PluginManager,
    _reset_plugin_managers_for_tests,
    get_plugin_manager,
    rediscover_plugins_if_changed,
)

KEY = "platforms/demo"


def _write_platform_plugin(base, version="0.1.0"):
    """A category plugin at ``<base>/platforms/demo`` (registry key ``platforms/demo``)."""
    plugin_dir = base / "platforms" / "demo"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({
        "name": "demo-platform",
        "label": "Demo",
        "kind": "platform",
        "version": version,
        "description": "discovery staleness probe",
    }))
    (plugin_dir / "__init__.py").write_text("def register(ctx):\n    pass\n")
    return plugin_dir


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "plugins").mkdir(parents=True)
    bundled = tmp_path / "bundled"
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [KEY]}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    _reset_plugin_managers_for_tests()
    yield home, bundled
    _reset_plugin_managers_for_tests()


def test_discovery_inputs_changed_tracks_override_add_and_remove(hermes_home):
    home, bundled = hermes_home
    _write_platform_plugin(bundled)
    manager = PluginManager()
    manager.discover_and_load()
    assert manager._plugins[KEY].manifest.source == "bundled"
    assert manager.discovery_inputs_changed() is False

    user_dir = _write_platform_plugin(home / "plugins")
    assert manager.discovery_inputs_changed() is True
    # The unforced sweep is the one the serve path used to run: it keeps the earlier registration.
    manager.discover_and_load()
    assert manager._plugins[KEY].manifest.source == "bundled"
    manager.discover_and_load(force=True)
    assert manager._plugins[KEY].manifest.source == "user"
    assert manager.discovery_inputs_changed() is False

    shutil.rmtree(user_dir.parent)  # the operator rolls the override back
    assert manager.discovery_inputs_changed() is True
    manager.discover_and_load(force=True)
    assert manager._plugins[KEY].manifest.source == "bundled"


def test_discovery_inputs_changed_tracks_a_version_bump(hermes_home):
    home, bundled = hermes_home
    _write_platform_plugin(bundled)
    manager = PluginManager()
    manager.discover_and_load()
    assert manager.discovery_inputs_changed() is False
    _write_platform_plugin(home / "plugins", version="0.2.0")
    assert manager.discovery_inputs_changed() is True


def test_discovery_inputs_changed_is_false_before_the_first_sweep(hermes_home):
    _home, bundled = hermes_home
    _write_platform_plugin(bundled)
    manager = get_plugin_manager()
    # Nothing recorded yet: the pending sweep reads the current directories anyway, so a caller must
    # not treat this as "changed" and force a second, redundant sweep.
    assert manager.discovery_inputs_changed() is False


def test_rediscover_plugins_if_changed_applies_a_new_override(hermes_home):
    home, bundled = hermes_home
    _write_platform_plugin(bundled)
    get_plugin_manager().discover_and_load()
    _write_platform_plugin(home / "plugins")

    assert rediscover_plugins_if_changed() is True
    assert get_plugin_manager()._plugins[KEY].manifest.source == "user"


def test_rediscover_plugins_if_changed_keeps_registrations_when_nothing_changed(hermes_home):
    _home, bundled = hermes_home
    _write_platform_plugin(bundled)
    manager = get_plugin_manager()
    manager.discover_and_load()
    loaded = manager._plugins[KEY]

    assert rediscover_plugins_if_changed() is False
    # Same loaded record: an unchanged source set must not unload and re-import every plugin on a
    # plain config/.env-driven re-serve.
    assert get_plugin_manager()._plugins[KEY] is loaded


def test_discovery_inputs_changed_tracks_a_gate_edit(hermes_home):
    home, bundled = hermes_home
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": []}}), encoding="utf-8")
    _write_platform_plugin(bundled)
    _write_platform_plugin(home / "plugins")  # override on disk, gated off
    manager = PluginManager()
    manager.discover_and_load()
    assert manager.discovery_inputs_changed() is False
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [KEY]}}), encoding="utf-8")
    # Hand-editing plugins.enabled must also be revalidated: otherwise a profile restart would keep
    # serving the gate verdict of a sweep made before the edit.
    assert manager.discovery_inputs_changed() is True
