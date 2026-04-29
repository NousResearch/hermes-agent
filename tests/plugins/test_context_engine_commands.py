"""Tests for slash commands registered by repo-shipped context engines."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch

from hermes_cli.plugins import PluginManager, get_plugin_command_handler


def _write_context_engine(root: Path, name: str, register_line: str) -> Path:
    sys.modules.pop(f"plugins.context_engine.{name}", None)
    engine_dir = root / name
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / "__init__.py").write_text(_context_engine_source(name, register_line))
    return engine_dir


def _context_engine_source(name: str, register_body: str) -> str:
    return (
        "from agent.context_engine import ContextEngine\n\n"
        "class StubEngine(ContextEngine):\n"
        "    @property\n"
        "    def name(self):\n"
        f"        return {name!r}\n\n"
        "    def update_from_response(self, usage):\n"
        "        return None\n\n"
        "    def should_compress(self, prompt_tokens=None):\n"
        "        return False\n\n"
        "    def compress(self, messages, current_tokens=None, focus_topic=None):\n"
        "        return messages\n\n"
        "def register(ctx):\n"
        "    ctx.register_context_engine(StubEngine())\n"
        f"    {register_body}\n"
    )


def _write_raw_context_engine(root: Path, name: str, source: str) -> Path:
    sys.modules.pop(f"plugins.context_engine.{name}", None)
    engine_dir = root / name
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / "__init__.py").write_text(source)
    return engine_dir


def _configure_context_engine(tmp_path, monkeypatch, name: str) -> None:
    hermes_home = tmp_path / "home"
    hermes_home.mkdir(exist_ok=True)
    (hermes_home / "config.yaml").write_text(f"context:\n  engine: {name}\n")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))


def test_repo_context_engine_loader_registers_slash_command(tmp_path, monkeypatch):
    """Repo-shipped context engines should expose registered slash commands."""
    import hermes_cli.plugins as plugins_mod
    import plugins.context_engine as context_engine_mod

    engine_root = tmp_path / "context_engine"
    _write_context_engine(
        engine_root,
        "slash_engine",
        'ctx.register_command("lcm", lambda raw_args: f"lcm:{raw_args}", description="LCM diagnostics")',
    )
    _configure_context_engine(tmp_path, monkeypatch, "slash_engine")
    monkeypatch.setattr(context_engine_mod, "_CONTEXT_ENGINE_PLUGINS_DIR", engine_root)
    manager = PluginManager()
    manager._discovered = True

    with patch.object(plugins_mod, "_plugin_manager", manager):
        engine = context_engine_mod.load_context_engine("slash_engine")
        assert engine is not None
        assert engine.name == "slash_engine"

        handler = get_plugin_command_handler("lcm")
        assert handler is not None
        assert handler("status") == "lcm:status"
        assert manager._plugin_commands["lcm"]["description"] == "LCM diagnostics"
        assert manager._plugin_commands["lcm"]["plugin"] == "context_engine:slash_engine"

        from hermes_cli.commands import gateway_help_lines, should_bypass_active_session, telegram_bot_commands

        assert ("lcm", "LCM diagnostics") in telegram_bot_commands()
        assert "`/lcm` -- LCM diagnostics" in gateway_help_lines()
        assert should_bypass_active_session("lcm") is True


def test_repo_context_engine_command_conflicts_with_builtins_are_rejected(tmp_path, monkeypatch, caplog):
    """Repo-shipped context-engine commands must keep built-in conflict protection."""
    import hermes_cli.plugins as plugins_mod
    import plugins.context_engine as context_engine_mod

    engine_root = tmp_path / "context_engine"
    _write_context_engine(
        engine_root,
        "conflict_engine",
        'ctx.register_command("help", lambda raw_args: "bad", description="Conflicting command")',
    )
    monkeypatch.setattr(context_engine_mod, "_CONTEXT_ENGINE_PLUGINS_DIR", engine_root)
    manager = PluginManager()
    manager._discovered = True

    with patch.object(plugins_mod, "_plugin_manager", manager):
        with caplog.at_level(logging.WARNING, logger="plugins.context_engine"):
            engine = context_engine_mod.load_context_engine("conflict_engine")

        assert engine is not None
        assert "help" not in manager._plugin_commands
        assert "conflicts with a built-in command" in caplog.text


def test_loading_different_context_engine_clears_stale_context_engine_commands(tmp_path, monkeypatch):
    """Switching active context engines should not leave stale slash commands behind."""
    import hermes_cli.plugins as plugins_mod
    import plugins.context_engine as context_engine_mod

    engine_root = tmp_path / "context_engine"
    _write_context_engine(
        engine_root,
        "slash_engine",
        'ctx.register_command("lcm", lambda raw_args: "old", description="Old")',
    )
    _write_context_engine(
        engine_root,
        "plain_engine",
        "pass",
    )
    monkeypatch.setattr(context_engine_mod, "_CONTEXT_ENGINE_PLUGINS_DIR", engine_root)
    manager = PluginManager()
    manager._discovered = True

    with patch.object(plugins_mod, "_plugin_manager", manager):
        assert context_engine_mod.load_context_engine("slash_engine") is not None
        assert "lcm" in manager._plugin_commands

        assert context_engine_mod.load_context_engine("plain_engine") is not None
        assert "lcm" not in manager._plugin_commands


def test_missing_context_engine_clears_stale_context_engine_commands(tmp_path, monkeypatch):
    """Failed context-engine lookups should not leave stale slash commands dispatchable."""
    import hermes_cli.plugins as plugins_mod
    import plugins.context_engine as context_engine_mod

    engine_root = tmp_path / "context_engine"
    _write_context_engine(
        engine_root,
        "slash_engine",
        'ctx.register_command("lcm", lambda raw_args: "old", description="Old")',
    )
    monkeypatch.setattr(context_engine_mod, "_CONTEXT_ENGINE_PLUGINS_DIR", engine_root)
    manager = PluginManager()
    manager._discovered = True

    with patch.object(plugins_mod, "_plugin_manager", manager):
        assert context_engine_mod.load_context_engine("slash_engine") is not None
        assert "lcm" in manager._plugin_commands

        assert context_engine_mod.load_context_engine("missing_engine") is None
        assert "lcm" not in manager._plugin_commands


def test_failed_context_engine_register_does_not_commit_pending_commands(tmp_path, monkeypatch):
    """Commands buffered before a register() crash should not become dispatchable."""
    import hermes_cli.plugins as plugins_mod
    import plugins.context_engine as context_engine_mod

    engine_root = tmp_path / "context_engine"
    _write_raw_context_engine(
        engine_root,
        "crashy_engine",
        _context_engine_source(
            "crashy_engine",
            'ctx.register_command("lcm", lambda raw_args: "bad", description="Bad")\n    raise RuntimeError("boom")',
        ),
    )
    monkeypatch.setattr(context_engine_mod, "_CONTEXT_ENGINE_PLUGINS_DIR", engine_root)
    manager = PluginManager()
    manager._discovered = True

    with patch.object(plugins_mod, "_plugin_manager", manager):
        engine = context_engine_mod.load_context_engine("crashy_engine")

        assert engine is not None
        assert engine.name == "crashy_engine"
        assert "lcm" not in manager._plugin_commands


def test_context_engine_register_without_engine_does_not_commit_pending_commands(tmp_path, monkeypatch):
    """A register() hook must register an engine before its slash commands become active."""
    import hermes_cli.plugins as plugins_mod
    import plugins.context_engine as context_engine_mod

    engine_root = tmp_path / "context_engine"
    _write_raw_context_engine(
        engine_root,
        "no_engine",
        "def register(ctx):\n"
        "    ctx.register_command('lcm', lambda raw_args: 'bad', description='Bad')\n",
    )
    monkeypatch.setattr(context_engine_mod, "_CONTEXT_ENGINE_PLUGINS_DIR", engine_root)
    manager = PluginManager()
    manager._discovered = True

    with patch.object(plugins_mod, "_plugin_manager", manager):
        assert context_engine_mod.load_context_engine("no_engine") is None
        assert "lcm" not in manager._plugin_commands


def test_context_engine_command_does_not_overwrite_normal_plugin_command(tmp_path, monkeypatch, caplog):
    """Context-engine commands must not silently replace normal plugin commands."""
    import hermes_cli.plugins as plugins_mod
    import plugins.context_engine as context_engine_mod

    engine_root = tmp_path / "context_engine"
    _write_context_engine(
        engine_root,
        "slash_engine",
        'ctx.register_command("lcm", lambda raw_args: "context", description="Context")',
    )
    monkeypatch.setattr(context_engine_mod, "_CONTEXT_ENGINE_PLUGINS_DIR", engine_root)
    manager = PluginManager()
    manager._discovered = True
    normal_handler = lambda raw_args: "normal"
    manager._plugin_commands["lcm"] = {
        "handler": normal_handler,
        "description": "Normal plugin command",
        "plugin": "normal-plugin",
        "args_hint": "",
    }

    with patch.object(plugins_mod, "_plugin_manager", manager):
        with caplog.at_level(logging.WARNING, logger="plugins.context_engine"):
            assert context_engine_mod.load_context_engine("slash_engine") is not None

        assert manager._plugin_commands["lcm"]["handler"] is normal_handler
        assert manager._plugin_commands["lcm"]["plugin"] == "normal-plugin"
        assert "already registered by plugin 'normal-plugin'" in caplog.text


def test_discovery_failure_does_not_clear_active_context_engine_commands(tmp_path, monkeypatch):
    """Availability discovery must stay side-effect-free even if a candidate crashes."""
    import hermes_cli.plugins as plugins_mod
    import plugins.context_engine as context_engine_mod

    engine_root = tmp_path / "context_engine"
    _write_raw_context_engine(
        engine_root,
        "crashy_discovery",
        _context_engine_source(
            "crashy_discovery",
            'ctx.register_command("other", lambda raw_args: "bad", description="Bad")\n    raise RuntimeError("boom")',
        ),
    )
    monkeypatch.setattr(context_engine_mod, "_CONTEXT_ENGINE_PLUGINS_DIR", engine_root)
    manager = PluginManager()
    manager._discovered = True
    active_handler = lambda raw_args: "active"
    manager._plugin_commands["lcm"] = {
        "handler": active_handler,
        "description": "Active context engine command",
        "plugin": "context_engine:lcm",
        "args_hint": "",
    }

    with patch.object(plugins_mod, "_plugin_manager", manager):
        discovered = context_engine_mod.discover_context_engines()

        assert discovered == [("crashy_discovery", "", True)]
        assert manager._plugin_commands["lcm"]["handler"] is active_handler
        assert "other" not in manager._plugin_commands


def test_fresh_gateway_surfaces_configured_context_engine_command_before_agent_init(tmp_path, monkeypatch):
    """Gateway command surfaces should preload commands for configured context engines."""
    import hermes_cli.plugins as plugins_mod
    import plugins.context_engine as context_engine_mod
    from hermes_cli.commands import gateway_help_lines, telegram_bot_commands

    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("context:\n  engine: slash_engine\n")
    engine_root = tmp_path / "context_engine"
    _write_context_engine(
        engine_root,
        "slash_engine",
        'ctx.register_command("lcm", lambda raw_args: f"fresh:{raw_args}", description="Fresh LCM")',
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(context_engine_mod, "_CONTEXT_ENGINE_PLUGINS_DIR", engine_root)
    manager = PluginManager()

    with patch.object(plugins_mod, "_plugin_manager", manager):
        assert "lcm" not in manager._plugin_commands

        assert ("lcm", "Fresh LCM") in telegram_bot_commands()
        assert "`/lcm` -- Fresh LCM" in gateway_help_lines()
        handler = get_plugin_command_handler("lcm")
        assert handler is not None
        assert handler("status") == "fresh:status"


def test_forced_plugin_refresh_resyncs_configured_context_engine_command(tmp_path, monkeypatch):
    """Forced plugin discovery should not leave the context-engine command sync marker stale."""
    import hermes_cli.plugins as plugins_mod
    import plugins.context_engine as context_engine_mod

    _configure_context_engine(tmp_path, monkeypatch, "slash_engine")
    engine_root = tmp_path / "context_engine"
    _write_context_engine(
        engine_root,
        "slash_engine",
        'ctx.register_command("lcm", lambda raw_args: f"fresh:{raw_args}", description="Fresh LCM")',
    )
    monkeypatch.setattr(context_engine_mod, "_CONTEXT_ENGINE_PLUGINS_DIR", engine_root)
    manager = PluginManager()

    with patch.object(plugins_mod, "_plugin_manager", manager):
        handler = get_plugin_command_handler("lcm")
        assert handler is not None
        assert handler("status") == "fresh:status"
        assert getattr(manager, "_context_engine_commands_synced_for") == "slash_engine"

        manager.discover_and_load(force=True)
        assert "lcm" not in manager._plugin_commands
        assert not hasattr(manager, "_context_engine_commands_synced_for")

        handler = get_plugin_command_handler("lcm")
        assert handler is not None
        assert handler("again") == "fresh:again"


def test_context_engine_discovery_does_not_surface_inactive_slash_commands(tmp_path, monkeypatch):
    """Availability discovery should not register commands for inactive engines."""
    import hermes_cli.plugins as plugins_mod
    import plugins.context_engine as context_engine_mod

    engine_root = tmp_path / "context_engine"
    _write_context_engine(
        engine_root,
        "inactive_engine",
        'ctx.register_command("inactive", lambda raw_args: "bad", description="Inactive")',
    )
    monkeypatch.setattr(context_engine_mod, "_CONTEXT_ENGINE_PLUGINS_DIR", engine_root)
    manager = PluginManager()
    manager._discovered = True

    with patch.object(plugins_mod, "_plugin_manager", manager):
        discovered = context_engine_mod.discover_context_engines()

        assert discovered == [("inactive_engine", "", True)]
        assert "inactive" not in manager._plugin_commands
