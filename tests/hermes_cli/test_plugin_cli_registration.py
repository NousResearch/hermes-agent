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
from unittest.mock import MagicMock

import pytest

from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
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

    def test_skips_plugin_without_register_cli(self, tmp_path, monkeypatch):
        """An active plugin with cli.py but no register_cli returns nothing."""
        plugin_dir = tmp_path / "noplugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("pass\n")
        (plugin_dir / "cli.py").write_text("def some_other_fn():\n    pass\n")

        import plugins.memory as pm
        original_dir = pm._MEMORY_PLUGINS_DIR
        monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", tmp_path)
        monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: "noplugin")
        try:
            cmds = pm.discover_plugin_cli_commands()
        finally:
            monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", original_dir)
            sys.modules.pop("plugins.memory.noplugin.cli", None)

        assert len(cmds) == 0

    def test_skips_plugin_without_cli_py(self, tmp_path, monkeypatch):
        """An active provider without cli.py returns nothing."""
        plugin_dir = tmp_path / "nocli"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("pass\n")

        import plugins.memory as pm
        original_dir = pm._MEMORY_PLUGINS_DIR
        monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", tmp_path)
        monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: "nocli")
        try:
            cmds = pm.discover_plugin_cli_commands()
        finally:
            monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", original_dir)

        assert len(cmds) == 0


# ── Honcho register_cli ──────────────────────────────────────────────────


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


# ── _surface_generic_plugin_cli_commands ─────────────────────────────────
#
# Covers the helper extracted from main()'s argparse setup that surfaces
# every plugin-registered CLI command as a top-level ``hermes <name>``
# subparser. Without this helper, ``register_cli_command()`` populates
# ``manager._cli_commands`` but nothing reads it for top-level argparse,
# so generic (non-memory) plugins are silently invisible.


class TestSurfaceGenericPluginCliCommands:
    def _build_parser(self):
        """Tiny argparse skeleton matching main()'s top-level shape."""
        parser = argparse.ArgumentParser(prog="hermes")
        subparsers = parser.add_subparsers(dest="command")
        return parser, subparsers

    def _make_manager_with_command(
        self,
        name: str = "mytool",
        setup_fn=None,
        handler_fn=None,
        help_text: str = "test help",
        description: str = "test description",
    ) -> PluginManager:
        """Return a PluginManager whose _cli_commands has one entry."""
        mgr = PluginManager()
        mgr._cli_commands[name] = {
            "name": name,
            "help": help_text,
            "description": description,
            "setup_fn": setup_fn or (lambda p: p.add_argument("--flag")),
            "handler_fn": handler_fn,
            "plugin": "test-plugin",
        }
        return mgr

    def test_adds_registered_command_as_top_level_subparser(self):
        """A plugin's register_cli_command-registered name becomes invocable."""
        from hermes_cli.main import _surface_generic_plugin_cli_commands

        mgr = self._make_manager_with_command(
            name="mytool",
            setup_fn=lambda p: p.add_argument("--flag"),
        )
        parser, subparsers = self._build_parser()

        _surface_generic_plugin_cli_commands(subparsers, mgr)

        ns = parser.parse_args(["mytool", "--flag", "value"])
        assert ns.command == "mytool"
        assert ns.flag == "value"

    def test_attaches_handler_via_set_defaults(self):
        """When handler_fn is provided it lands as args.func."""
        from hermes_cli.main import _surface_generic_plugin_cli_commands

        handler = MagicMock()
        mgr = self._make_manager_with_command(
            name="dohandle",
            setup_fn=lambda p: p.add_argument("arg"),
            handler_fn=handler,
        )
        parser, subparsers = self._build_parser()

        _surface_generic_plugin_cli_commands(subparsers, mgr)

        ns = parser.parse_args(["dohandle", "VAL"])
        assert ns.func is handler
        assert ns.arg == "VAL"

    def test_no_handler_means_no_func_default(self):
        """A None handler_fn must not set args.func (caller dispatches manually)."""
        from hermes_cli.main import _surface_generic_plugin_cli_commands

        mgr = self._make_manager_with_command(
            name="nofunc",
            setup_fn=lambda p: None,
            handler_fn=None,
        )
        parser, subparsers = self._build_parser()

        _surface_generic_plugin_cli_commands(subparsers, mgr)

        ns = parser.parse_args(["nofunc"])
        assert ns.command == "nofunc"
        assert not hasattr(ns, "func")

    def test_skips_names_already_in_subparsers(self):
        """A plugin can't override a built-in subcommand by re-registering its name."""
        from hermes_cli.main import _surface_generic_plugin_cli_commands

        mgr = self._make_manager_with_command(
            name="chat",  # would conflict with the built-in `hermes chat`
            setup_fn=lambda p: p.add_argument("--should-not-add"),
            handler_fn=MagicMock(),
        )
        parser, subparsers = self._build_parser()

        # Pre-register `chat` as a built-in-like subparser
        chat_parser = subparsers.add_parser("chat", help="real chat")
        chat_parser.add_argument("--real-flag")
        sentinel = object()
        chat_parser.set_defaults(func=sentinel)

        _surface_generic_plugin_cli_commands(subparsers, mgr)

        # Real `chat` still works
        ns = parser.parse_args(["chat", "--real-flag", "yes"])
        assert ns.real_flag == "yes"
        assert ns.func is sentinel

        # The plugin's `--should-not-add` was never added to `chat`
        with pytest.raises(SystemExit):
            parser.parse_args(["chat", "--should-not-add", "x"])

    def test_iterates_multiple_registered_commands(self):
        """Every entry in _cli_commands is surfaced (not just the first)."""
        from hermes_cli.main import _surface_generic_plugin_cli_commands

        mgr = PluginManager()
        for name in ("alpha", "beta", "gamma"):
            mgr._cli_commands[name] = {
                "name": name,
                "help": f"{name} help",
                "description": "",
                "setup_fn": lambda p, n=name: p.add_argument(f"--{n}-flag"),
                "handler_fn": None,
                "plugin": "multi-plugin",
            }
        parser, subparsers = self._build_parser()

        _surface_generic_plugin_cli_commands(subparsers, mgr)

        for name in ("alpha", "beta", "gamma"):
            ns = parser.parse_args([name, f"--{name}-flag", "v"])
            assert ns.command == name

    def test_empty_registry_is_a_noop(self):
        """An empty _cli_commands dict adds nothing and does not raise."""
        from hermes_cli.main import _surface_generic_plugin_cli_commands

        mgr = PluginManager()  # no _cli_commands entries
        parser, subparsers = self._build_parser()

        _surface_generic_plugin_cli_commands(subparsers, mgr)

        assert subparsers.choices == {}
