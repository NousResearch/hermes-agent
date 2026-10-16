"""Memory plugin CLI discovery: only the active provider's ``cli.py`` commands are exposed."""

import sys


# ── Memory plugin CLI discovery ───────────────────────────────────────────


class TestMemoryPluginCliDiscovery:
    def test_discovers_active_plugin_with_register_cli(self, tmp_path, monkeypatch):
        """Only the active memory provider's CLI commands are discovered."""
        plugin_dir = tmp_path / "testplugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("pass\n")
        (plugin_dir / "cli.py").write_text(
            "def register_cli(subparser):\n"
            "    subparser.add_argument('--test')\n"
            "\n"
            "def testplugin_command(args):\n"
            "    pass\n"
        )
        (plugin_dir / "plugin.yaml").write_text(
            "name: testplugin\ndescription: A test plugin\n"
        )

        # Also create a second plugin that should NOT be discovered
        other_dir = tmp_path / "otherplugin"
        other_dir.mkdir()
        (other_dir / "__init__.py").write_text("pass\n")
        (other_dir / "cli.py").write_text(
            "def register_cli(subparser):\n"
            "    subparser.add_argument('--other')\n"
        )

        import plugins.memory as pm
        original_dir = pm._MEMORY_PLUGINS_DIR
        mod_key = "plugins.memory.testplugin.cli"
        sys.modules.pop(mod_key, None)

        monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", tmp_path)
        # Set testplugin as the active provider
        monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: "testplugin")
        try:
            cmds = pm.discover_plugin_cli_commands()
        finally:
            monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", original_dir)
            sys.modules.pop(mod_key, None)

        # Only testplugin should be discovered, not otherplugin
        assert len(cmds) == 1
        assert cmds[0]["name"] == "testplugin"
        assert cmds[0]["help"] == "A test plugin"
        assert callable(cmds[0]["setup_fn"])
        assert cmds[0]["handler_fn"].__name__ == "testplugin_command"

    def test_returns_nothing_when_no_active_provider(self, tmp_path, monkeypatch):
        """No commands when memory.provider is not set in config."""
        plugin_dir = tmp_path / "testplugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("pass\n")
        (plugin_dir / "cli.py").write_text(
            "def register_cli(subparser):\n    pass\n"
        )

        import plugins.memory as pm
        original_dir = pm._MEMORY_PLUGINS_DIR
        monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", tmp_path)
        monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: None)
        try:
            cmds = pm.discover_plugin_cli_commands()
        finally:
            monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", original_dir)

        assert len(cmds) == 0
