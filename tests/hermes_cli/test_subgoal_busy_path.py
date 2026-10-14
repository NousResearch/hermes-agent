"""Regression tests for classic-CLI mid-run goal-control dispatch.

Goal controls typed while the agent was running used to wait in
``self._pending_input`` until ``self.chat()`` finished. These tests exercise
the detector and inline TUI dispatch without starting a prompt_toolkit app.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_cli():
    """Create a HermesCLI instance with prompt_toolkit stubbed out."""
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            _cli_mod.__dict__, {"CLI_CONFIG": _clean_config}
        ):
            return _cli_mod.HermesCLI()


class TestGoalControlInlineDetector:
    def test_detects_subgoal_when_agent_running(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_goal_control_command_inline(
            "/subgoal also verify the rollback path") is True

    def test_detects_bare_subgoal_when_agent_running(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_goal_control_command_inline("/subgoal") is True

    def test_ignores_subgoal_when_agent_idle(self):
        cli = _make_cli()
        cli._agent_running = False
        assert cli._should_handle_goal_control_command_inline(
            "/subgoal also verify the rollback path") is False

    def test_ignores_non_slash_input(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_goal_control_command_inline(
            "subgoal without slash") is False
        assert cli._should_handle_goal_control_command_inline("") is False

    def test_ignores_other_slash_commands(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_goal_control_command_inline("/queue hello") is False
        assert cli._should_handle_goal_control_command_inline("/stop") is False
        assert cli._should_handle_goal_control_command_inline("/help") is False
        assert cli._should_handle_goal_control_command_inline("/loop stop") is False

    def test_ignores_subgoal_with_attached_images(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_goal_control_command_inline(
            "/subgoal text", has_images=True) is False

    def test_detects_goal_control_verbs_when_agent_running(self):
        """Mirrors the gateway whitelist: state-only /goal verbs dispatch mid-run."""
        cli = _make_cli()
        cli._agent_running = True
        for arg in ("status", "show", "pause", "resume", "clear", "stop", "done", "unwait",
                    "wait 42", "gate list", "gate add make check"):
            assert cli._should_handle_goal_control_command_inline(f"/goal {arg}") is True, arg

    def test_rejects_new_goal_text_when_agent_running(self):
        """Setting a NEW goal mid-run stays queued — mirrors the gateway's _busy_goal_command."""
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_goal_control_command_inline(
            "/goal ship the release") is False
        assert cli._should_handle_goal_control_command_inline(
            "/goal draft new text") is False

    def test_bare_goal_is_control(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_goal_control_command_inline("/goal") is True


class TestGoalControlBusyPathDispatch:
    def test_subgoal_runs_mid_turn_not_queued(self):
        cli = _make_cli()
        cli._agent_running = True
        cli._pending_input = MagicMock()
        captured: dict = {}

        def _fake_handler(cmd_original):
            captured["cmd"] = cmd_original
            return None

        cli._handle_subgoal_command = _fake_handler

        assert cli._tui_enter_inline_command(None, "/subgoal also check metrics", False) is True
        assert captured["cmd"] == "/subgoal also check metrics"
        cli._pending_input.put.assert_not_called()

    def test_goal_control_runs_mid_turn(self):
        cli = _make_cli()
        cli._agent_running = True
        cli._pending_input = MagicMock()

        calls: list = []
        cli._handle_goal_command = lambda cmd: calls.append(cmd)

        assert cli._tui_enter_inline_command(None, "/goal pause", False) is True
        assert calls == ["/goal pause"]
        cli._pending_input.put.assert_not_called()

    def test_new_goal_text_still_queues(self):
        """/goal <new text> mid-run must NOT dispatch inline — it queues for next turn."""
        cli = _make_cli()
        cli._agent_running = True
        cli._pending_input = MagicMock()

        goal_calls: list = []
        cli._handle_goal_command = lambda cmd: goal_calls.append(cmd)

        assert cli._tui_enter_inline_command(None, "/goal new objective", False) is False
        assert goal_calls == []

    def test_idle_subgoal_falls_through(self):
        """Idle-path /subgoal returns False → normal queue → process_loop dispatch."""
        cli = _make_cli()
        cli._agent_running = False
        assert cli._tui_enter_inline_command(None, "/subgoal extra criterion", False) is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])