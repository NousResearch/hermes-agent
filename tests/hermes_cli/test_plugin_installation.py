"""The installation lock serializes plugin writers per home, and discovery waits for it."""
from __future__ import annotations

import threading

import yaml


def test_the_lock_excludes_a_concurrent_installation_writer(tmp_path, monkeypatch):
    from hermes_cli.plugin_installation import plugin_installation_lock
    from hermes_cli.plugins_cmd import _set_plugin_enabled

    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    done = threading.Event()

    def contender():
        _set_plugin_enabled("contender", enable=True)
        done.set()

    thread = threading.Thread(target=contender, daemon=True)
    with plugin_installation_lock(home):
        _set_plugin_enabled("owner", enable=True)  # same-thread nesting
        thread.start()
        assert not done.wait(1), "a writer escaped the installation lock"
    thread.join(15)
    assert done.is_set()
    assert yaml.safe_load((home / "config.yaml").read_text())["plugins"]["enabled"] == ["contender", "owner"]


def test_discovery_waits_for_the_installation_and_then_sees_it(tmp_path, monkeypatch):
    from hermes_cli import plugins
    from hermes_cli.plugin_installation import plugin_installation_lock

    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: tmp_path / "no-bundled")
    (home / "config.yaml").write_text("plugins:\n  enabled: [late_probe]\n")
    manager = plugins.get_plugin_manager()
    monkeypatch.setattr(manager, "_scan_entry_points", lambda: [])
    with plugin_installation_lock(home):
        plugins.start_background_plugin_discovery()
        worker = plugins._background_discovery_thread
        worker.join(1)
        assert worker.is_alive() and not manager._discovered, "discovery ran while the installation was in flight"
        plugin = home / "plugins" / "late_probe"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: late_probe\n")
        (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n")
    worker.join(15)
    assert manager._discovered
    assert [p["name"] for p in manager.list_plugins()] == ["late_probe"]
