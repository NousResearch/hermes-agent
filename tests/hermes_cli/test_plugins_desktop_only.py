"""Desktop-only plugins carry no Python half and must not be reported as failed loads.

Coverage for #132741, split from ``test_plugins.py`` (FILE_LINES ratchet). The loader
skip and the warning it removes are described in
``PluginLoaderMixin._is_desktop_only_plugin``.
"""

import logging

import hermes_yaml as yaml

from hermes_cli.plugins import PluginManager


class TestDesktopOnlyPluginLoading:
    """A plugin whose whole payload is ``desktop/plugin.js`` has no Python half: the Desktop app
    loads that half itself, so the loader has nothing to import and no ``register()`` to call. Importing
    the absent ``__init__.py`` anyway logged ``Failed to load plugin '<name>': No __init__.py in ...``
    on every load pass — per session creation, not only per process start — in front of the real load
    failures an operator is reading the log for (#132741)."""

    @staticmethod
    def _plugin(tmp_path, name, *, files, enabled=True):
        hermes_home = tmp_path / "hermes_test"
        plugin_dir = hermes_home / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": name}), encoding="utf-8")
        for rel, body in files.items():
            target = plugin_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        if enabled:
            (hermes_home / "config.yaml").write_text(
                yaml.safe_dump({"plugins": {"enabled": [name]}}), encoding="utf-8")
        return hermes_home, plugin_dir

    def test_desktop_only_plugin_loads_without_a_warning(self, tmp_path, monkeypatch, caplog):
        hermes_home, _ = self._plugin(tmp_path, "agent-log", files={
            "desktop/plugin.js": "export default { render() {} }\n",
        })
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        mgr = PluginManager()
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            mgr.discover_and_load()

        entry = mgr._plugins["agent-log"]
        assert entry.error is None
        assert entry.enabled
        assert entry.module is None
        assert "Failed to load plugin" not in caplog.text
        assert "No __init__.py" not in caplog.text

    def test_directory_with_python_but_no_init_still_warns(self, tmp_path, monkeypatch, caplog):
        """The warning is worth reading, so the skip must not swallow a plugin whose Python half is
        missing for a real reason."""
        hermes_home, _ = self._plugin(tmp_path, "half-written", files={
            "tools.py": "def register_tools(ctx):\n    pass\n",
        })
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        mgr = PluginManager()
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            mgr.discover_and_load()

        assert "Failed to load plugin" in caplog.text
        assert "No __init__.py" in caplog.text
        assert not mgr._plugins["half-written"].enabled

    def test_desktop_half_alongside_python_still_runs_register(self, tmp_path, monkeypatch):
        """A package that ships both halves is an ordinary Python plugin: the skip is keyed on the
        absent ``__init__.py``, not on the presence of ``desktop/``."""
        hermes_home, plugin_dir = self._plugin(tmp_path, "hermes-media", files={
            "desktop/plugin.js": "export default { render() {} }\n",
            "__init__.py": (
                "from pathlib import Path\n"
                "Path(__file__).with_name('imported').write_text('yes', encoding='utf-8')\n"
                "def register(ctx):\n"
                "    pass\n"
            ),
        })
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert (plugin_dir / "imported").exists()
        entry = mgr._plugins["hermes-media"]
        assert entry.enabled and entry.error is None
