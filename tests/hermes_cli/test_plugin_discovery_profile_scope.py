"""Background discovery must publish the selected profile's plugin cache, not the launcher's."""

import json
from pathlib import Path

import pytest

from hermes_cli import plugins
from hermes_cli.config import atomic_config_write
from hermes_cli.plugins_loader import _plugin_home_scope
from hermes_constants import get_hermes_home


_PLUGIN = '''
def register(ctx):
    ctx.register_tool(
        name="discovery_scope_probe",
        toolset="discovery_scope_probe_set",
        schema={"name": "discovery_scope_probe", "description": "Fixture", "parameters": {}},
        handler=lambda args, **kw: "{}",
    )
'''


@pytest.mark.parametrize("selected", ["launch", "secondary"])
def test_background_discovery_publishes_own_profile_cache(tmp_path, monkeypatch, selected):
    homes = {name: tmp_path / name for name in ("launch", "secondary")}
    for home in homes.values():
        home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(homes["launch"]))
    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: tmp_path / "no-bundled")
    monkeypatch.setattr(plugins.PluginManager, "_scan_entry_points", lambda self: [])
    monkeypatch.setattr(plugins, "_background_discovery_thread", None)
    plugins._reset_plugin_managers_for_tests()

    home = homes[selected]
    directory = home / "plugins" / "discovery-probe"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text("name: discovery-probe\nversion: 0.1.0\n", encoding="utf-8")
    (directory / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
    atomic_config_write(home / "config.yaml", {"plugins": {"enabled": ["discovery-probe"]}})
    worker = None
    try:
        with _plugin_home_scope(home):
            manager = plugins.get_plugin_manager()
            plugins.start_background_plugin_discovery()
            worker = plugins._background_discovery_thread
        assert get_hermes_home() == homes["launch"]
        assert worker is not None
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert manager._plugins["discovery-probe"].enabled
        cache = home / "cache" / "plugin_toolset_keys.json"
        assert cache.exists(), f"discovery did not publish the owning profile cache: {home}"
        assert json.loads(cache.read_text(encoding="utf-8"))["toolset_keys"] == ["discovery_scope_probe_set"]
        other = homes["secondary" if selected == "launch" else "launch"]
        assert not (other / "cache" / "plugin_toolset_keys.json").exists()
    finally:
        if worker is not None:
            worker.join(timeout=10)
            assert not worker.is_alive(), "discovery must stop before its managers are unloaded"
        plugins._reset_plugin_managers_for_tests()
