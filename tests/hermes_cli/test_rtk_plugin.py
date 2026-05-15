"""Tests for the RTK plugin and pre_tool_call arg-override mechanism.

TDD red-phase: these tests define the desired behaviour before implementation.
"""

import json
import os
import subprocess
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


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


# ── 1. RTK Plugin Unit Tests ──────────────────────────────────────────────


class TestRTKPlugin:
    """Tests for the RTK rewrite plugin itself."""

    def test_rtk_rewrite_returns_modified_command(self, monkeypatch):
        """RTK plugin hook callback returns rewrite directive with rewritten command."""
        from hermes_plugins.rtk import _rewrite_terminal_command

        monkeypatch.delenv("HERMES_RTK_DISABLE", raising=False)
        with patch("shutil.which", return_value="/usr/bin/rtk"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="rtk git status", stderr=""
            )
            result = _rewrite_terminal_command(
                tool_name="terminal",
                args={"command": "git status"},
            )
        assert result is not None
        assert result["action"] == "rewrite"
        assert "args" in result
        assert "command" in result["args"]
        # The rewritten command should start with "rtk"
        assert result["args"]["command"].startswith("rtk ")

    def test_rtk_rewrite_skips_non_terminal_tools(self):
        """RTK plugin ignores non-terminal tool calls."""
        from hermes_plugins.rtk import _rewrite_terminal_command

        result = _rewrite_terminal_command(
            tool_name="read_file",
            args={"path": "/tmp/test.txt"},
        )
        assert result is None

    def test_rtk_rewrite_skips_when_rtk_not_available(self):
        """RTK plugin returns None when rtk binary is not found."""
        from hermes_plugins.rtk import _rewrite_terminal_command

        with patch("shutil.which", return_value=None):
            result = _rewrite_terminal_command(
                tool_name="terminal",
                args={"command": "git status"},
            )
        assert result is None

    def test_rtk_rewrite_skips_unsupported_commands(self):
        """RTK plugin returns None when rtk rewrite exits 1 (unsupported command)."""
        from hermes_plugins.rtk import _rewrite_terminal_command

        with (
            patch("shutil.which", return_value="/usr/bin/rtk"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr=""
            )
            result = _rewrite_terminal_command(
                tool_name="terminal",
                args={"command": "echo hello"},
            )
        assert result is None

    def test_rtk_rewrite_handles_timeout(self):
        """RTK plugin returns None if rtk rewrite times out."""
        from hermes_plugins.rtk import _rewrite_terminal_command

        with (
            patch("shutil.which", return_value="/usr/bin/rtk"),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="rtk", timeout=2)),
        ):
            result = _rewrite_terminal_command(
                tool_name="terminal",
                args={"command": "git status"},
            )
        assert result is None

    def test_rtk_rewrite_preserves_command_on_no_change(self, monkeypatch):
        """If rtk rewrite returns the same command, plugin still returns the rewrite
        directive (the prefix 'rtk' is the savings mechanism)."""
        from hermes_plugins.rtk import _rewrite_terminal_command

        monkeypatch.delenv("HERMES_RTK_DISABLE", raising=False)
        with patch("shutil.which", return_value="/usr/bin/rtk"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="rtk git status", stderr=""
            )
            result = _rewrite_terminal_command(
                tool_name="terminal",
                args={"command": "git status"},
            )
        assert result is not None
        assert result["args"]["command"] == "rtk git status"

    def test_rtk_rewrite_skips_when_command_missing(self):
        """RTK plugin returns None when args dict has no 'command' key."""
        from hermes_plugins.rtk import _rewrite_terminal_command

        result = _rewrite_terminal_command(
            tool_name="terminal",
            args={},
        )
        assert result is None

    def test_rtk_rewrite_env_disable(self, monkeypatch):
        """RTK plugin is disabled when HERMES_RTK_DISABLE is set."""
        from hermes_plugins.rtk import _rewrite_terminal_command

        for value in ("1", "true", "yes", "on"):
            monkeypatch.setenv("HERMES_RTK_DISABLE", value)
            result = _rewrite_terminal_command(
                tool_name="terminal",
                args={"command": "git status"},
            )
            assert result is None
            monkeypatch.delenv("HERMES_RTK_DISABLE", raising=False)

    def test_rtk_rewrite_env_disable_false_values_do_not_disable(self, monkeypatch):
        """Falsy env values should not disable RTK rewriting."""
        from hermes_plugins.rtk import _rewrite_terminal_command

        monkeypatch.setenv("HERMES_RTK_DISABLE", "0")
        with (
            patch("shutil.which", return_value="/usr/bin/rtk"),
            patch(
                "subprocess.run",
                return_value=MagicMock(returncode=0, stdout="rtk git status", stderr=""),
            ),
        ):
            result = _rewrite_terminal_command(
                tool_name="terminal",
                args={"command": "git status"},
            )
        assert result is not None


# ── 2. Core: get_pre_tool_call_arg_overrides ──────────────────────────────


class TestPreToolCallArgOverrides:
    """Tests for the arg-override extraction from pre_tool_call hook results."""

    def test_rewrite_directive_collected(self):
        """A hook returning {"action": "rewrite", "args": {...}} is collected."""
        from hermes_cli.plugins import get_pre_tool_call_arg_overrides

        hook_results = [
            {"action": "rewrite", "args": {"command": "rtk git status"}},
        ]
        overrides = get_pre_tool_call_arg_overrides(hook_results)
        assert overrides == {"command": "rtk git status"}

    def test_block_directive_not_collected(self):
        """A hook returning {"action": "block", ...} is not collected as override."""
        from hermes_cli.plugins import get_pre_tool_call_arg_overrides

        hook_results = [
            {"action": "block", "message": "forbidden"},
        ]
        overrides = get_pre_tool_call_arg_overrides(hook_results)
        assert overrides == {}

    def test_none_returns_ignored(self):
        """Hook callbacks returning None are skipped."""
        from hermes_cli.plugins import get_pre_tool_call_arg_overrides

        hook_results = [None, {"action": "rewrite", "args": {"x": "y"}}]
        overrides = get_pre_tool_call_arg_overrides(hook_results)
        assert overrides == {"x": "y"}

    def test_non_dict_returns_ignored(self):
        """Hook callbacks returning non-dict values are skipped."""
        from hermes_cli.plugins import get_pre_tool_call_arg_overrides

        hook_results = ["some string", 42, {"action": "rewrite", "args": {"z": "1"}}]
        overrides = get_pre_tool_call_arg_overrides(hook_results)
        assert overrides == {"z": "1"}

    def test_multiple_rewrite_directives_last_wins(self):
        """Multiple rewrite directives: last one wins (simple merge, last key wins)."""
        from hermes_cli.plugins import get_pre_tool_call_arg_overrides

        hook_results = [
            {"action": "rewrite", "args": {"command": "rtk git log"}},
            {"action": "rewrite", "args": {"command": "rtk git status"}},
        ]
        overrides = get_pre_tool_call_arg_overrides(hook_results)
        assert overrides == {"command": "rtk git status"}

    def test_empty_hook_results(self):
        """Empty hook results produce empty overrides."""
        from hermes_cli.plugins import get_pre_tool_call_arg_overrides

        overrides = get_pre_tool_call_arg_overrides([])
        assert overrides == {}


# ── 3. Core: get_pre_tool_call_block_message returns hook_results ────────


class TestBlockMessageReturnsHookResults:
    """get_pre_tool_call_block_message should return (block_message, hook_results)
    so callers can reuse the hook results for arg overrides without double-firing."""

    def test_returns_tuple_when_no_block(self):
        """When no block directive, returns (None, hook_results)."""
        from hermes_cli.plugins import get_pre_tool_call_block_message

        # We need a live PluginManager, so patch invoke_hook
        with patch("hermes_cli.plugins.invoke_hook", return_value=[
            {"action": "rewrite", "args": {"command": "rtk git status"}}
        ]):
            result = get_pre_tool_call_block_message(
                "terminal", {"command": "git status"},
            )
        assert isinstance(result, tuple)
        block_msg, hook_results = result
        assert block_msg is None
        assert len(hook_results) == 1
        assert hook_results[0]["action"] == "rewrite"

    def test_returns_tuple_with_block(self):
        """When block directive found, returns (message, hook_results)."""
        from hermes_cli.plugins import get_pre_tool_call_block_message

        with patch("hermes_cli.plugins.invoke_hook", return_value=[
            {"action": "block", "message": "forbidden"},
        ]):
            result = get_pre_tool_call_block_message(
                "terminal", {"command": "rm -rf /"},
            )
        assert isinstance(result, tuple)
        block_msg, hook_results = result
        assert block_msg == "forbidden"
        assert len(hook_results) == 1


# ── 4. Integration: handle_function_call applies arg overrides ───────────


class TestHandleFunctionCallArgMutation:
    """handle_function_call should apply arg overrides from pre_tool_call hooks."""

    def test_terminal_command_rewritten_by_plugin(self):
        """When a pre_tool_call hook returns a rewrite, the terminal command is
        rewritten before dispatch."""
        from model_tools import handle_function_call

        with (
            patch("model_tools.registry.dispatch", return_value='{"ok":true}') as mock_dispatch,
            patch("hermes_cli.plugins.get_pre_tool_call_directives",
                  return_value=(None, {"command": "rtk git status"})),
        ):
            result = handle_function_call(
                "terminal", {"command": "git status"},
                task_id="t1",
            )
        assert result == '{"ok":true}'
        # The dispatched args should have the rewritten command
        dispatched_args = mock_dispatch.call_args[0][1]
        assert dispatched_args["command"] == "rtk git status"

    def test_no_rewrite_when_hook_returns_nothing(self):
        """When hooks return nothing, original args are passed through."""
        from model_tools import handle_function_call

        with (
            patch("model_tools.registry.dispatch", return_value='{"ok":true}') as mock_dispatch,
            patch("hermes_cli.plugins.get_pre_tool_call_directives",
                  return_value=(None, None)),
        ):
            result = handle_function_call(
                "terminal", {"command": "git status"},
                task_id="t1",
            )
        dispatched_args = mock_dispatch.call_args[0][1]
        assert dispatched_args["command"] == "git status"

    def test_block_takes_precedence_over_rewrite(self):
        """When a block directive and a rewrite directive both appear,
        block wins and the function returns an error."""
        from model_tools import handle_function_call

        with (
            patch("model_tools.registry.dispatch", return_value='{"ok":true}') as mock_dispatch,
            patch("hermes_cli.plugins.get_pre_tool_call_directives",
                  return_value=("forbidden", {"command": "rtk rm -rf /"})),
        ):
            result = handle_function_call(
                "terminal", {"command": "rm -rf /"},
                task_id="t1",
            )
        parsed = json.loads(result)
        assert "error" in parsed
        mock_dispatch.assert_not_called()


# ── 5. Plugin discovery ──────────────────────────────────────────────────


class TestRTKPluginDiscovery:
    """The RTK plugin directory is discovered and loaded by the plugin system."""

    def test_rtk_plugin_loads(self, tmp_path, monkeypatch):
        """The RTK plugin can be discovered and loaded."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "rtk",
            register_body='ctx.register_hook("pre_tool_call", lambda **kw: None)',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        from hermes_cli.plugins import PluginManager, _get_enabled_plugins
        monkeypatch.setattr("hermes_cli.plugins._get_enabled_plugins", lambda: {"rtk"})

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "rtk" in mgr._plugins
        assert mgr._plugins["rtk"].enabled
