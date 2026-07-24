from pathlib import Path

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager, PluginManifest


def _fake_platform_manifest(tmp_path: Path) -> PluginManifest:
    plugin_dir = tmp_path / "plugins" / "platforms" / "fake"
    plugin_dir.mkdir(parents=True)
    return PluginManifest(
        name="fake-platform",
        key="platforms/fake",
        kind="platform",
        source="bundled",
        path=str(plugin_dir),
    )


def test_explicitly_enabled_bundled_platform_loads_for_cli(monkeypatch, tmp_path):
    manifest = _fake_platform_manifest(tmp_path)
    manager = PluginManager()
    loaded = []
    deferred = []

    def fake_scan_directory(path, source, skip_names=None):
        if Path(path).name == "platforms":
            return [manifest]
        return []

    monkeypatch.setattr(manager, "_scan_directory", fake_scan_directory)
    monkeypatch.setattr(manager, "_scan_entry_points", lambda: [])
    monkeypatch.setattr(manager, "_load_plugin", lambda m: loaded.append(m.key))
    monkeypatch.setattr(manager, "_register_deferred_platform", lambda m: deferred.append(m.key))
    monkeypatch.setattr(plugins_mod, "_get_enabled_plugins", lambda: {"platforms/fake"})
    monkeypatch.setattr(plugins_mod, "_get_disabled_plugins", lambda: set())

    manager.discover_and_load(force=True)

    assert loaded == ["platforms/fake"]
    assert deferred == []


def test_non_enabled_bundled_platform_still_defers(monkeypatch, tmp_path):
    manifest = _fake_platform_manifest(tmp_path)
    manager = PluginManager()
    loaded = []
    deferred = []

    def fake_scan_directory(path, source, skip_names=None):
        if Path(path).name == "platforms":
            return [manifest]
        return []

    monkeypatch.setattr(manager, "_scan_directory", fake_scan_directory)
    monkeypatch.setattr(manager, "_scan_entry_points", lambda: [])
    monkeypatch.setattr(manager, "_load_plugin", lambda m: loaded.append(m.key))
    monkeypatch.setattr(manager, "_register_deferred_platform", lambda m: deferred.append(m.key))
    monkeypatch.setattr(plugins_mod, "_get_enabled_plugins", lambda: set())
    monkeypatch.setattr(plugins_mod, "_get_disabled_plugins", lambda: set())

    manager.discover_and_load(force=True)

    assert loaded == []
    assert deferred == ["platforms/fake"]
