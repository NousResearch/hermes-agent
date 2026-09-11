"""Tests for plugin CLI registration system.

Covers:
  - PluginContext.register_cli_command()
  - PluginManager._cli_commands storage
  - get_plugin_cli_commands() convenience function
  - Memory plugin CLI discovery (discover_plugin_cli_commands)
  - Honcho register_cli() builds correct argparse tree
"""

import argparse
import sys
from unittest.mock import MagicMock

import pytest

from hermes_cli import plugins
from hermes_cli.plugin_invocation import PluginInvocationContext
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
        assert entry["availability"] is None

    def test_overwrites_on_duplicate(self):
        ctx, mgr = self._make_ctx()
        ctx.register_cli_command("x", "first", MagicMock())
        ctx.register_cli_command("x", "second", MagicMock())
        assert mgr._cli_commands["x"]["help"] == "second"

    def test_availability_filters_catalog_and_handler(self, monkeypatch):
        ctx, mgr = self._make_ctx()
        handler = MagicMock()
        ctx.register_cli_command(
            "guarded",
            "Guarded command",
            MagicMock(),
            handler,
            availability=lambda invocation: (
                invocation.platform == "cli"
                and invocation.authenticated_actor == "nas-user-7"
            ),
        )
        monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: mgr)
        allowed = PluginInvocationContext(
            "default", None, "cli", "nas-user-7", None, None, None, None, "root"
        )
        denied = PluginInvocationContext(
            "default", None, "cli", None, None, None, None, None, "root"
        )

        assert "guarded" in plugins.get_plugin_cli_commands(allowed)
        assert plugins.get_plugin_cli_command_handler("guarded", allowed) is handler
        assert plugins.get_plugin_cli_commands(denied) == {}
        assert plugins.get_plugin_cli_command_handler("guarded", denied) is None

    def test_non_callable_availability_is_rejected(self):
        ctx, _mgr = self._make_ctx()

        with pytest.raises(TypeError, match="CLI command availability must be callable"):
            ctx.register_cli_command(
                "guarded", "Guarded command", MagicMock(), availability=True
            )

    def test_parser_handler_binds_context_and_rechecks_availability(
        self, monkeypatch
    ):
        from hermes_cli import main as main_mod
        from hermes_cli import plugin_invocation as invocation_mod

        ctx, mgr = self._make_ctx()
        enabled = [True]
        seen = []

        def handler(_args):
            invocation = ctx.invocation
            seen.append((invocation.profile, invocation.authenticated_actor))
            return 7

        ctx.register_cli_command(
            "guarded",
            "Guarded command",
            lambda parser: parser.add_argument("--flag", action="store_true"),
            handler,
            availability=lambda invocation: enabled[0]
            and invocation.authenticated_actor == "nas-user-7",
        )
        monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: mgr)

        def invocation():
            return PluginInvocationContext(
                "work", None, "cli", "nas-user-7", None, None, None, None, "root"
            )

        monkeypatch.setattr(invocation_mod, "_new_local_plugin_invocation", lambda **_kw: invocation())
        parser = argparse.ArgumentParser(prog="hermes")
        subparsers = parser.add_subparsers(dest="command")
        main_mod._attach_plugin_cli_command(
            subparsers, mgr._cli_commands["guarded"], trusted_context=True
        )
        args = parser.parse_args(["guarded", "--flag"])

        assert main_mod._run_cli_handler(args, parser) == 7
        assert seen == [("work", "nas-user-7")]

        enabled[0] = False
        with pytest.raises(SystemExit) as denied:
            main_mod._run_cli_handler(args, parser)
        assert denied.value.code == 2
        assert seen == [("work", "nas-user-7")]

    @pytest.mark.parametrize("root_handler", [False, True])
    def test_setup_installed_nested_handler_is_gated_after_parse(
        self, monkeypatch, root_handler
    ):
        from hermes_cli import main as main_mod
        from hermes_cli import plugin_invocation as invocation_mod

        ctx, mgr = self._make_ctx()
        enabled = [True]
        seen = []

        def nested_handler(_args):
            seen.append(ctx.invocation.platform)
            return 9

        def setup(parser):
            if not root_handler:
                parser.set_defaults(func=lambda _args: 3)
            nested = parser.add_subparsers(dest="action").add_parser("nested")
            nested.set_defaults(func=nested_handler)

        ctx.register_cli_command(
            "guarded",
            "Guarded command",
            setup,
            (lambda _args: 4) if root_handler else None,
            availability=lambda _invocation: enabled[0],
        )
        monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: mgr)
        monkeypatch.setattr(
            invocation_mod,
            "_new_local_plugin_invocation",
            lambda **_kw: PluginInvocationContext(
                "work", None, "cli", None, None, None, None, None, "root"
            ),
        )
        parser = argparse.ArgumentParser(prog="hermes")
        subparsers = parser.add_subparsers(dest="command")
        main_mod._attach_plugin_cli_command(
            subparsers, mgr._cli_commands["guarded"], trusted_context=True
        )
        args = parser.parse_args(["guarded", "nested"])

        assert main_mod._run_cli_handler(args, parser) == 9
        assert seen == ["cli"]

        enabled[0] = False
        with pytest.raises(SystemExit) as denied:
            main_mod._run_cli_handler(args, parser)
        assert denied.value.code == 2
        assert seen == ["cli"]

    def test_setup_only_root_handler_is_gated_after_parse(self, monkeypatch):
        from hermes_cli import main as main_mod
        from hermes_cli import plugin_invocation as invocation_mod

        ctx, mgr = self._make_ctx()
        enabled = [True]
        seen = []

        def setup(parser):
            parser.set_defaults(func=lambda _args: seen.append(ctx.invocation.platform))

        ctx.register_cli_command(
            "guarded",
            "Guarded command",
            setup,
            availability=lambda _invocation: enabled[0],
        )
        monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: mgr)
        monkeypatch.setattr(
            invocation_mod,
            "_new_local_plugin_invocation",
            lambda **_kw: PluginInvocationContext(
                "work", None, "cli", None, None, None, None, None, "root"
            ),
        )
        parser = argparse.ArgumentParser(prog="hermes")
        subparsers = parser.add_subparsers(dest="command")
        main_mod._attach_plugin_cli_command(
            subparsers, mgr._cli_commands["guarded"], trusted_context=True
        )
        args = parser.parse_args(["guarded"])

        main_mod._run_cli_handler(args, parser)
        assert seen == ["cli"]

        enabled[0] = False
        with pytest.raises(SystemExit) as denied:
            main_mod._run_cli_handler(args, parser)
        assert denied.value.code == 2
        assert seen == ["cli"]

    def test_unavailable_registered_command_is_not_attached(self, monkeypatch):
        from hermes_cli import main as main_mod
        from hermes_cli import plugin_invocation as invocation_mod
        from plugins import memory

        ctx, mgr = self._make_ctx()
        handler = MagicMock()
        ctx.register_cli_command(
            "guarded",
            "Guarded command",
            lambda _parser: None,
            handler,
            availability=lambda _invocation: False,
        )
        monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: mgr)
        monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
        monkeypatch.setattr(memory, "discover_plugin_cli_commands", lambda: [])
        monkeypatch.setattr(main_mod, "_resolve_deferred_platform_cli_command", lambda _name: None)
        monkeypatch.setattr(
            invocation_mod,
            "_new_local_plugin_invocation",
            lambda **_kw: PluginInvocationContext(
                "work", None, "cli", None, None, None, None, None, "root"
            ),
        )
        parser = argparse.ArgumentParser(prog="hermes")
        subparsers = parser.add_subparsers(dest="command")

        main_mod._register_plugin_cli_commands(subparsers)

        assert "guarded" not in subparsers.choices
        with pytest.raises(SystemExit) as denied:
            parser.parse_args(["guarded"])
        assert denied.value.code == 2
        handler.assert_not_called()


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


# ── Honcho register_cli ──────────────────────────────────────────────────


# ── ProviderCollector no-op ──────────────────────────────────────────────


class TestProviderCollectorCliNoop:
    def test_register_cli_command_is_noop(self):
        """_ProviderCollector.register_cli_command is a no-op (doesn't crash)."""
        from plugins.memory import _ProviderCollector

        collector = _ProviderCollector("test-provider")
        collector.register_cli_command(
            name="test", help="test", setup_fn=lambda s: None
        )
        # Should not store anything — CLI is discovered via file convention
        assert not hasattr(collector, "_cli_commands")
