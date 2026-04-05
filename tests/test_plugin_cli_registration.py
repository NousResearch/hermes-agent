"""Tests for plugin CLI registration system.

Covers:
  - PluginContext.register_cli_command()
  - PluginManager._cli_commands storage
  - get_plugin_cli_commands() convenience function
  - Memory plugin CLI discovery (discover_plugin_cli_commands)
  - Honcho register_cli() builds correct argparse tree
"""

import argparse
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
    get_plugin_cli_commands,
)


# ── PluginContext.register_cli_command ─────────────────────────────────────


class TestRegisterCliCommand:
    def _make_ctx(self):
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin")
        return PluginContext(manifest, mgr), mgr

    def test_registers_command(self):
        ctx, mgr = self._make_ctx()
        setup = MagicMock()
        handler = MagicMock()
        ctx.register_cli_command(
            name="mycmd",
            help="Do something",
            setup_fn=setup,
            handler_fn=handler,
            description="Full description",
        )
        assert "mycmd" in mgr._cli_commands
        entry = mgr._cli_commands["mycmd"]
        assert entry["name"] == "mycmd"
        assert entry["help"] == "Do something"
        assert entry["setup_fn"] is setup
        assert entry["handler_fn"] is handler
        assert entry["plugin"] == "test-plugin"

    def test_overwrites_on_duplicate(self):
        ctx, mgr = self._make_ctx()
        ctx.register_cli_command("x", "first", MagicMock())
        ctx.register_cli_command("x", "second", MagicMock())
        assert mgr._cli_commands["x"]["help"] == "second"

    def test_handler_optional(self):
        ctx, mgr = self._make_ctx()
        ctx.register_cli_command("nocb", "test", MagicMock())
        assert mgr._cli_commands["nocb"]["handler_fn"] is None


class TestGetPluginCliCommands:
    def test_returns_dict(self):
        mgr = PluginManager()
        mgr._cli_commands["foo"] = {"name": "foo", "help": "bar"}
        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            cmds = get_plugin_cli_commands()
        assert cmds == {"foo": {"name": "foo", "help": "bar"}}
        # Top-level is a copy — adding to result doesn't affect manager
        cmds["new"] = {"name": "new"}
        assert "new" not in mgr._cli_commands


# ── Memory plugin CLI discovery ───────────────────────────────────────────


class TestMemoryPluginCliDiscovery:
    def test_discovers_plugin_with_register_cli(self, tmp_path, monkeypatch):
        """A memory plugin dir with cli.py containing register_cli is discovered."""
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

        # Patch _MEMORY_PLUGINS_DIR to our tmp dir
        import plugins.memory as pm
        original_dir = pm._MEMORY_PLUGINS_DIR

        # Clear any cached module to force reimport
        mod_key = "plugins.memory.testplugin.cli"
        sys.modules.pop(mod_key, None)

        monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", tmp_path)
        try:
            cmds = pm.discover_plugin_cli_commands()
        finally:
            monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", original_dir)
            sys.modules.pop(mod_key, None)

        assert len(cmds) == 1
        assert cmds[0]["name"] == "testplugin"
        assert cmds[0]["help"] == "A test plugin"
        assert callable(cmds[0]["setup_fn"])
        assert cmds[0]["handler_fn"].__name__ == "testplugin_command"

    def test_skips_plugin_without_register_cli(self, tmp_path, monkeypatch):
        """A memory plugin with cli.py but no register_cli is skipped."""
        plugin_dir = tmp_path / "noplugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("pass\n")
        (plugin_dir / "cli.py").write_text("def some_other_fn():\n    pass\n")

        import plugins.memory as pm
        original_dir = pm._MEMORY_PLUGINS_DIR
        monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", tmp_path)
        try:
            cmds = pm.discover_plugin_cli_commands()
        finally:
            monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", original_dir)
            sys.modules.pop("plugins.memory.noplugin.cli", None)

        assert len(cmds) == 0

    def test_skips_plugin_without_cli_py(self, tmp_path, monkeypatch):
        """A memory plugin dir without cli.py is skipped."""
        plugin_dir = tmp_path / "nocli"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("pass\n")

        import plugins.memory as pm
        original_dir = pm._MEMORY_PLUGINS_DIR
        monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", tmp_path)
        try:
            cmds = pm.discover_plugin_cli_commands()
        finally:
            monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", original_dir)

        assert len(cmds) == 0


# ── Honcho register_cli ──────────────────────────────────────────────────


class TestHonchoRegisterCli:
    def test_builds_subcommand_tree(self):
        """register_cli creates the expected subparser tree."""
        from plugins.memory.honcho.cli import register_cli

        parser = argparse.ArgumentParser()
        register_cli(parser)

        # Verify key subcommands exist by parsing them
        args = parser.parse_args(["status"])
        assert args.honcho_command == "status"

        args = parser.parse_args(["peer", "--user", "alice"])
        assert args.honcho_command == "peer"
        assert args.user == "alice"

        args = parser.parse_args(["mode", "tools"])
        assert args.honcho_command == "mode"
        assert args.mode == "tools"

        args = parser.parse_args(["tokens", "--context", "500"])
        assert args.honcho_command == "tokens"
        assert args.context == 500

        args = parser.parse_args(["--target-profile", "coder", "status"])
        assert args.target_profile == "coder"
        assert args.honcho_command == "status"

    def test_setup_redirects_to_memory_setup(self):
        """hermes honcho setup redirects to memory setup."""
        from plugins.memory.honcho.cli import register_cli

        parser = argparse.ArgumentParser()
        register_cli(parser)
        args = parser.parse_args(["setup"])
        assert args.honcho_command == "setup"

    def test_mode_choices_are_recall_modes(self):
        """Mode subcommand uses recall mode choices (hybrid/context/tools)."""
        from plugins.memory.honcho.cli import register_cli

        parser = argparse.ArgumentParser()
        register_cli(parser)

        # Valid recall modes should parse
        for mode in ("hybrid", "context", "tools"):
            args = parser.parse_args(["mode", mode])
            assert args.mode == mode

        # Old memoryMode values should fail
        with pytest.raises(SystemExit):
            parser.parse_args(["mode", "honcho"])


# ── ProviderCollector no-op ──────────────────────────────────────────────


class TestProviderCollectorCliNoop:
    def test_register_cli_command_is_noop(self):
        """_ProviderCollector.register_cli_command is a no-op (doesn't crash)."""
        from plugins.memory import _ProviderCollector

        collector = _ProviderCollector()
        collector.register_cli_command(
            name="test", help="test", setup_fn=lambda s: None
        )
        # Should not store anything — CLI is discovered via file convention
        assert not hasattr(collector, "_cli_commands")
