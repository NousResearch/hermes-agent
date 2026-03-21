"""Tests for the Hermes plugin system (hermes_cli.plugins)."""

import logging
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_cli.plugins import (
    ENTRY_POINTS_GROUP,
    VALID_HOOKS,
    LoadedPlugin,
    PluginContext,
    PluginManager,
    PluginManifest,
    get_plugin_manager,
    get_plugin_tool_names,
    get_plugin_command_handler,
    discover_plugins,
    invoke_hook,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_plugin_dir(base: Path, name: str, *, register_body: str = "pass",
                     manifest_extra: dict | None = None) -> Path:
    """Create a minimal plugin directory with plugin.yaml + __init__.py."""
    plugin_dir = base / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"name": name, "version": "0.1.0", "description": f"Test plugin {name}"}
    if manifest_extra:
        manifest.update(manifest_extra)

    (plugin_dir / "plugin.yaml").write_text(yaml.dump(manifest))
    (plugin_dir / "__init__.py").write_text(
        f"def register(ctx):\n    {register_body}\n"
    )
    return plugin_dir


# ── TestPluginDiscovery ────────────────────────────────────────────────────


class TestPluginDiscovery:
    """Tests for plugin discovery from directories and entry points."""

    def test_discover_user_plugins(self, tmp_path, monkeypatch):
        """Plugins in ~/.hermes/plugins/ are discovered."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(plugins_dir, "hello_plugin")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "hello_plugin" in mgr._plugins
        assert mgr._plugins["hello_plugin"].enabled

    def test_discover_project_plugins(self, tmp_path, monkeypatch):
        """Plugins in ./.hermes/plugins/ are discovered."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "true")
        plugins_dir = project_dir / ".hermes" / "plugins"
        _make_plugin_dir(plugins_dir, "proj_plugin")

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "proj_plugin" in mgr._plugins
        assert mgr._plugins["proj_plugin"].enabled

    def test_discover_project_plugins_skipped_by_default(self, tmp_path, monkeypatch):
        """Project plugins are not discovered unless explicitly enabled."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        plugins_dir = project_dir / ".hermes" / "plugins"
        _make_plugin_dir(plugins_dir, "proj_plugin")

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "proj_plugin" not in mgr._plugins

    def test_discover_is_idempotent(self, tmp_path, monkeypatch):
        """Calling discover_and_load() twice does not duplicate plugins."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(plugins_dir, "once_plugin")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()
        mgr.discover_and_load()  # second call should no-op

        assert len(mgr._plugins) == 1

    def test_discover_skips_dir_without_manifest(self, tmp_path, monkeypatch):
        """Directories without plugin.yaml are silently skipped."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        (plugins_dir / "no_manifest").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert len(mgr._plugins) == 0

    def test_entry_points_scanned(self, tmp_path, monkeypatch):
        """Entry-point based plugins are discovered (mocked)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        fake_module = types.ModuleType("fake_ep_plugin")
        fake_module.register = lambda ctx: None  # type: ignore[attr-defined]

        fake_ep = MagicMock()
        fake_ep.name = "ep_plugin"
        fake_ep.value = "fake_ep_plugin:register"
        fake_ep.group = ENTRY_POINTS_GROUP
        fake_ep.load.return_value = fake_module

        def fake_entry_points():
            result = MagicMock()
            result.select = MagicMock(return_value=[fake_ep])
            return result

        with patch("importlib.metadata.entry_points", fake_entry_points):
            mgr = PluginManager()
            mgr.discover_and_load()

        assert "ep_plugin" in mgr._plugins


# ── TestPluginLoading ──────────────────────────────────────────────────────


class TestPluginLoading:
    """Tests for plugin module loading."""

    def test_load_missing_init(self, tmp_path, monkeypatch):
        """Plugin dir without __init__.py records an error."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "bad_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "bad_plugin"}))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "bad_plugin" in mgr._plugins
        assert not mgr._plugins["bad_plugin"].enabled
        assert mgr._plugins["bad_plugin"].error is not None

    def test_load_missing_register_fn(self, tmp_path, monkeypatch):
        """Plugin without register() function records an error."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "no_reg"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "no_reg"}))
        (plugin_dir / "__init__.py").write_text("# no register function\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "no_reg" in mgr._plugins
        assert not mgr._plugins["no_reg"].enabled
        assert "no register()" in mgr._plugins["no_reg"].error

    def test_load_registers_namespace_module(self, tmp_path, monkeypatch):
        """Directory plugins are importable under hermes_plugins.<name>."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(plugins_dir, "ns_plugin")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        # Clean up any prior namespace module
        sys.modules.pop("hermes_plugins.ns_plugin", None)

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "hermes_plugins.ns_plugin" in sys.modules


# ── TestPluginHooks ────────────────────────────────────────────────────────


class TestPluginHooks:
    """Tests for lifecycle hook registration and invocation."""

    def test_register_and_invoke_hook(self, tmp_path, monkeypatch):
        """Registered hooks are called on invoke_hook()."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "hook_plugin",
            register_body='ctx.register_hook("pre_tool_call", lambda **kw: None)',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        # Should not raise
        mgr.invoke_hook("pre_tool_call", tool_name="test", args={}, task_id="t1")

    def test_hook_exception_does_not_propagate(self, tmp_path, monkeypatch):
        """A hook callback that raises does NOT crash the caller."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "bad_hook",
            register_body='ctx.register_hook("post_tool_call", lambda **kw: 1/0)',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        # Should not raise despite 1/0
        mgr.invoke_hook("post_tool_call", tool_name="x", args={}, result="r", task_id="")

    def test_invalid_hook_name_warns(self, tmp_path, monkeypatch, caplog):
        """Registering an unknown hook name logs a warning."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "warn_plugin",
            register_body='ctx.register_hook("on_banana", lambda **kw: None)',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            mgr = PluginManager()
            mgr.discover_and_load()

        assert any("on_banana" in record.message for record in caplog.records)


# ── TestPluginContext ──────────────────────────────────────────────────────


class TestPluginContext:
    """Tests for the PluginContext facade."""

    def test_register_tool_adds_to_registry(self, tmp_path, monkeypatch):
        """PluginContext.register_tool() puts the tool in the global registry."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "tool_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "tool_plugin"}))
        (plugin_dir / "__init__.py").write_text(
            'def register(ctx):\n'
            '    ctx.register_tool(\n'
            '        name="plugin_echo",\n'
            '        toolset="plugin_tool_plugin",\n'
            '        schema={"name": "plugin_echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}},\n'
            '        handler=lambda args, **kw: "echo",\n'
            '    )\n'
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "plugin_echo" in mgr._plugin_tool_names

        from tools.registry import registry
        assert "plugin_echo" in registry._tools


# ── TestPluginToolVisibility ───────────────────────────────────────────────


class TestPluginToolVisibility:
    """Plugin-registered tools appear in get_tool_definitions()."""

    def test_plugin_tools_in_definitions(self, tmp_path, monkeypatch):
        """Tools from plugins bypass the toolset filter."""
        import hermes_cli.plugins as plugins_mod

        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "vis_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "vis_plugin"}))
        (plugin_dir / "__init__.py").write_text(
            'def register(ctx):\n'
            '    ctx.register_tool(\n'
            '        name="vis_tool",\n'
            '        toolset="plugin_vis_plugin",\n'
            '        schema={"name": "vis_tool", "description": "Visible", "parameters": {"type": "object", "properties": {}}},\n'
            '        handler=lambda args, **kw: "ok",\n'
            '    )\n'
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()
        monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)

        from model_tools import get_tool_definitions
        tools = get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)
        tool_names = [t["function"]["name"] for t in tools]
        assert "vis_tool" in tool_names


# ── TestPluginManagerList ──────────────────────────────────────────────────


class TestPluginManagerList:
    """Tests for PluginManager.list_plugins()."""

    def test_list_empty(self):
        """Empty manager returns empty list."""
        mgr = PluginManager()
        assert mgr.list_plugins() == []

    def test_list_returns_sorted(self, tmp_path, monkeypatch):
        """list_plugins() returns results sorted by name."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(plugins_dir, "zulu")
        _make_plugin_dir(plugins_dir, "alpha")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        listing = mgr.list_plugins()
        names = [p["name"] for p in listing]
        assert names == sorted(names)

    def test_list_with_plugins(self, tmp_path, monkeypatch):
        """list_plugins() returns info dicts for each discovered plugin."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(plugins_dir, "alpha")
        _make_plugin_dir(plugins_dir, "beta")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        listing = mgr.list_plugins()
        names = [p["name"] for p in listing]
        assert "alpha" in names
        assert "beta" in names
        for p in listing:
            assert "enabled" in p
            assert "tools" in p
            assert "hooks" in p
            assert "commands" in p


# ── TestPluginCommands ────────────────────────────────────────────────────


class TestPluginCommands:
    """Tests for plugin slash command registration."""

    def test_register_command_adds_to_registry(self, tmp_path, monkeypatch):
        """PluginContext.register_command() adds a CommandDef to COMMAND_REGISTRY."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "cmd_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "cmd_plugin"}))
        (plugin_dir / "__init__.py").write_text(
            'def _greet(args):\n'
            '    return f"Hello, {args or \'world\'}!"\n'
            '\n'
            'def register(ctx):\n'
            '    ctx.register_command(\n'
            '        name="greet",\n'
            '        handler=_greet,\n'
            '        description="Greet someone",\n'
            '        args_hint="[name]",\n'
            '        aliases=("hi",),\n'
            '    )\n'
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        # Command handler is registered
        assert "greet" in mgr._plugin_commands
        assert "hi" in mgr._plugin_commands
        assert mgr._plugin_commands["greet"]("Alice") == "Hello, Alice!"
        assert mgr._plugin_commands["greet"]("") == "Hello, world!"

        # CommandDef is in the registry
        from hermes_cli.commands import resolve_command
        cmd_def = resolve_command("greet")
        assert cmd_def is not None
        assert cmd_def.name == "greet"
        assert cmd_def.description == "Greet someone"
        assert cmd_def.category == "Plugins"
        assert "hi" in cmd_def.aliases

        # Alias resolves to same CommandDef
        assert resolve_command("hi") is cmd_def

    def test_register_command_appears_in_help(self, tmp_path, monkeypatch):
        """Plugin commands appear in COMMANDS dict for /help display."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "help_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "help_plugin"}))
        (plugin_dir / "__init__.py").write_text(
            'def register(ctx):\n'
            '    ctx.register_command(\n'
            '        name="myhelpcmd",\n'
            '        handler=lambda args: "ok",\n'
            '        description="My help command",\n'
            '    )\n'
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        from hermes_cli.commands import COMMANDS, COMMANDS_BY_CATEGORY
        assert "/myhelpcmd" in COMMANDS
        assert "Plugins" in COMMANDS_BY_CATEGORY
        assert "/myhelpcmd" in COMMANDS_BY_CATEGORY["Plugins"]

    def test_register_command_tracks_on_loaded_plugin(self, tmp_path, monkeypatch):
        """LoadedPlugin.commands_registered tracks plugin commands."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "tracked_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "tracked_plugin"}))
        (plugin_dir / "__init__.py").write_text(
            'def register(ctx):\n'
            '    ctx.register_command(\n'
            '        name="tracked",\n'
            '        handler=lambda args: "ok",\n'
            '        aliases=("tr",),\n'
            '    )\n'
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        loaded = mgr._plugins["tracked_plugin"]
        assert "tracked" in loaded.commands_registered
        assert "tr" in loaded.commands_registered

    def test_get_plugin_command_handler(self, tmp_path, monkeypatch):
        """get_plugin_command_handler() returns handler or None."""
        import hermes_cli.plugins as plugins_mod

        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "handler_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "handler_plugin"}))
        (plugin_dir / "__init__.py").write_text(
            'def register(ctx):\n'
            '    ctx.register_command(\n'
            '        name="dostuff",\n'
            '        handler=lambda args: "did stuff",\n'
            '    )\n'
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()
        monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)

        handler = get_plugin_command_handler("dostuff")
        assert handler is not None
        assert handler("") == "did stuff"

        assert get_plugin_command_handler("nonexistent") is None

    def test_gateway_known_commands_updated(self, tmp_path, monkeypatch):
        """Plugin commands appear in GATEWAY_KNOWN_COMMANDS for gateway dispatch."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "gw_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "gw_plugin"}))
        (plugin_dir / "__init__.py").write_text(
            'def register(ctx):\n'
            '    ctx.register_command(\n'
            '        name="gwcmd",\n'
            '        handler=lambda args: "gw ok",\n'
            '    )\n'
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        from hermes_cli import commands as cmd_mod
        assert "gwcmd" in cmd_mod.GATEWAY_KNOWN_COMMANDS
