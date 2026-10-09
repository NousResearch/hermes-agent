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

    def test_rtk_rewrite_returns_changed_command(self, monkeypatch):
        """A rewritten command produces the request-middleware directive."""
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

    def test_rtk_rewrite_preserves_command_on_no_change(self, monkeypatch):
        """An identical command is a pass-through, not a rewrite directive."""
        from hermes_plugins.rtk import _rewrite_terminal_command

        monkeypatch.delenv("HERMES_RTK_DISABLE", raising=False)
        with (
            patch("shutil.which", return_value="/usr/bin/rtk"),
            patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="git status", stderr="")),
        ):
            assert _rewrite_terminal_command("terminal", {"command": "git status"}) is None

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

        with (
            patch("shutil.which", return_value="/usr/bin/rtk"),
            patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="rtk git status", stderr="")) as mock_run,
        ):
            for value in ("1", "true", "yes", "on"):
                monkeypatch.setenv("HERMES_RTK_DISABLE", value)
                result = _rewrite_terminal_command(
                    tool_name="terminal",
                    args={"command": "git status"},
                )
                assert result is None
                mock_run.assert_not_called()
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


# ── 2. Rewrites enter the request middleware before authorization ─────────


class TestRewriteMiddleware:
    def test_preserves_non_command_arguments(self):
        from hermes_plugins.rtk import _rewrite_terminal_command

        with patch("shutil.which", return_value="/usr/bin/rtk"), patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout="rtk git status", stderr=""),
        ):
            result = _rewrite_terminal_command(
                "terminal", {"command": "git status", "timeout": 30, "cwd": "/tmp"}
            )
        assert result["args"] == {
            "command": "rtk git status", "timeout": 30, "cwd": "/tmp"
        }


# ── 4. Integration: handle_function_call applies arg overrides ───────────


class TestHandleFunctionCallArgMutation:
    """Request rewrites remain visible to authorization before dispatch."""

    @pytest.mark.parametrize("approved", [True, False])
    def test_rewritten_command_requires_human_approval(self, approved):
        from model_tools import handle_function_call

        seen = []
        def hooks(name, **kwargs):
            if name == "pre_tool_call":
                seen.append(kwargs["args"])
                return [{"action": "approve", "message": "review", "rule_key": "terminal:test"}]
            return []

        with (
            patch("model_tools.registry.dispatch", return_value='{"ok":true}') as dispatch,
            patch("hermes_cli.plugins.has_middleware", return_value=True),
            patch("hermes_cli.plugins.invoke_middleware", return_value=[
                {"args": {"command": "rtk git status", "timeout": 30}}
            ]),
            patch("hermes_cli.plugins.invoke_hook", side_effect=hooks),
            patch("tools.approval.request_tool_approval", return_value={
                "approved": approved, "message": "denied"
            }) as approval,
        ):
            result = handle_function_call("terminal", {"command": "git status", "timeout": 30})
        assert seen == [{"command": "rtk git status", "timeout": 30}]
        approval.assert_called_once()
        if approved:
            dispatch.assert_called_once()
            assert dispatch.call_args.args[1] == seen[0]
            assert result == '{"ok":true}'
        else:
            dispatch.assert_not_called()
            assert "denied" in result

    def test_approval_failure_does_not_dispatch_rewritten_command(self):
        from model_tools import handle_function_call

        def hooks(name, **kwargs):
            return [{"action": "approve", "message": "review"}] if name == "pre_tool_call" else []

        with (
            patch("model_tools.registry.dispatch") as dispatch,
            patch("hermes_cli.plugins.has_middleware", return_value=True),
            patch("hermes_cli.plugins.invoke_middleware", return_value=[{"args": {"command": "rtk ls"}}]),
            patch("hermes_cli.plugins.invoke_hook", side_effect=hooks),
            patch("tools.approval.request_tool_approval", side_effect=RuntimeError("unavailable")),
        ):
            result = handle_function_call("terminal", {"command": "ls"})
        dispatch.assert_not_called()
        assert "approval gate failed" in result


# ── 5. Plugin discovery ──────────────────────────────────────────────────


class TestRTKPluginDiscovery:
    """The RTK plugin directory is discovered and loaded by the plugin system."""

    def test_rtk_plugin_loads(self, tmp_path, monkeypatch):
        """The RTK plugin can be discovered and loaded."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "rtk_discovery_probe",
            register_body='ctx.register_hook("pre_tool_call", lambda **kw: None)',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        from hermes_cli.plugins import PluginManager, _get_enabled_plugins
        monkeypatch.setattr("hermes_cli.plugins._get_enabled_plugins", lambda: {"rtk_discovery_probe"})

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "rtk_discovery_probe" in mgr._plugins
        assert mgr._plugins["rtk_discovery_probe"].enabled


def test_rtk_registers_request_middleware():
    from hermes_plugins import rtk

    ctx = MagicMock()
    rtk.register(ctx)
    ctx.register_middleware.assert_called_once_with("tool_request", rtk._rewrite_terminal_command)
    ctx.register_hook.assert_not_called()
