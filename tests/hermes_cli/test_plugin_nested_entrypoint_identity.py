"""Directory identity wins over its same-name packaged entry point."""
from importlib.metadata import EntryPoint, EntryPoints
import sys
from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_cmd import _discover_all_plugins


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mode", ["enabled", "disabled", "raises"])
def test_directory_identity_prevents_duplicate_registration(tmp_path, monkeypatch, nested, mode):
    home = tmp_path / "home"
    root = home / "plugins" / "demo"
    package = root / "demo_pkg" if nested else root
    package.mkdir(parents=True)
    marker = tmp_path / "registrations.txt"
    body = (
        "from pathlib import Path\n"
        "def register(ctx):\n"
        f"    with Path({str(marker)!r}).open('a') as f: f.write(ctx.plugin_id + '\\n')\n"
    )
    if mode == "raises":
        body += "    raise RuntimeError('fixture registration failure')\n"
    (package / "__init__.py").write_text(body, encoding="utf-8")
    (package / "plugin.yaml").write_text("name: demo\nversion: 1.0.0\n", encoding="utf-8")
    key = "demo/demo_pkg" if nested else "demo"
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["demo"], "disabled": [key] if mode == "disabled" else []
    }}), encoding="utf-8")
    empty = tmp_path / "bundled"
    empty.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    monkeypatch.syspath_prepend(str(root if nested else root.parent))
    module_name = "demo_pkg" if nested else "demo"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    ep = EntryPoint(name="demo", value=module_name, group="hermes_agent.plugins")
    monkeypatch.setattr("importlib.metadata.entry_points", lambda: EntryPoints([ep]))
    from hermes_cli.config import load_config
    assert load_config()["plugins"]["enabled"] == ["demo"]
    manager = PluginManager()
    try:
        manager.discover_and_load()
        rows = [row for row in _discover_all_plugins() if row[0] == "demo"]
        assert [(row[3], row[5]) for row in rows] == [("user", key)]
        assert set(manager._plugins) == {key}
        assert Path(manager._plugins[key].manifest.path) == package
        assert manager._plugins[key].enabled is (mode == "enabled"), manager._plugins[key].error
        if mode == "disabled":
            assert not marker.exists()
        else:
            assert marker.read_text().splitlines() == [key]
        if mode == "raises":
            assert "fixture registration failure" in manager._plugins[key].error
        assert module_name not in sys.modules  # never import the installed alias
    finally:
        manager.unload()
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("has_entrypoint", [False, True])
def test_entrypoint_without_directory_remains_available(tmp_path, monkeypatch, has_entrypoint):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: [solo]\n", encoding="utf-8")
    empty = tmp_path / "bundled"
    empty.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    package = tmp_path / "solo_pkg.py"
    package.write_text("def register(ctx):\n    ctx.register_hook('on_session_start', lambda **kw: 'solo')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    ep = EntryPoint(name="solo", value="solo_pkg", group="hermes_agent.plugins")
    monkeypatch.setattr("importlib.metadata.entry_points", lambda: EntryPoints([ep] if has_entrypoint else []))
    manager = PluginManager()
    try:
        manager.discover_and_load()
        rows = _discover_all_plugins()
        assert set(manager._plugins) == ({"solo"} if has_entrypoint else set())
        assert [row[3] for row in rows] == (["entrypoint"] if has_entrypoint else [])
        if has_entrypoint:
            assert manager._plugins["solo"].enabled
            assert manager.invoke_hook("on_session_start") == ["solo"]
    finally:
        manager.unload()
        sys.modules.pop("solo_pkg", None)
