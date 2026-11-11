"""A plugin's event-bus subscriptions end with the plugin: a forced re-discovery (the gateway's
reload-plugins verb) delivers each event once, and a disabled plugin receives none."""
from pathlib import Path

import hermes_yaml as yaml


def _write_config(home: Path, enabled: list[str]) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": enabled}}), encoding="utf-8")


def test_reload_and_disable_end_event_subscriptions(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManifest

    home = tmp_path / "home"
    plugin = home / "plugins" / "listener"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(yaml.safe_dump({"name": "listener", "version": "1.0"}), encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "from pathlib import Path\n"
        "LOG = Path(__file__).with_name('deliveries')\n"
        "def _on_ping(**payload):\n"
        "    with LOG.open('a') as fh:\n"
        "        fh.write('x')\n"
        "def register(ctx):\n"
        "    ctx.subscribe('emitter:ping', _on_ping)\n", encoding="utf-8")
    deliveries = plugin / "deliveries"
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    _write_config(home, ["listener"])

    plugins_mod.discover_plugins()
    plugins_mod.discover_plugins(force=True)
    plugins_mod.discover_plugins(force=True)
    manager = plugins_mod.get_plugin_manager()
    emitter = PluginContext(PluginManifest(name="emitter", source="user", key="emitter"), manager)
    emitter.emit("ping")
    assert manager._wait_for_event_dispatch()
    assert deliveries.read_text() == "x"

    _write_config(home, [])
    plugins_mod.discover_plugins(force=True)
    deliveries.unlink()
    assert emitter.emit("ping") == 0
    assert manager._wait_for_event_dispatch()
    assert not deliveries.exists()
