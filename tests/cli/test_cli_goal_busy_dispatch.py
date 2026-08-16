"""Regression tests for classic-CLI mid-run /goal gate and goal-control dispatch (#87446).

Background
----------
When a user sets busy_input_mode="queue" and enters `/goal gate add ...` or other
goal control subcommands (/goal wait, /goal unwait, /goal pause, /goal resume,
/goal show, /goal status, /subgoal) while a goal turn is running, the command
was previously queued into `_pending_input` instead of being dispatched immediately.
Because `_pending_input` is only processed after the goal loop or turn finishes,
the goal is often completed or inactive by the time `/goal gate add` runs,
surfacing as `/goal gate add: no active goal` and preventing quality gates
from attaching to the active goal.

These tests exercise `_should_handle_goal_command_inline` across all goal control
commands, new-goal text prompts, and idle/busy states.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch


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


class TestGoalInlineDetector:
    def test_detects_goal_gate_commands_when_agent_running(self):
        cli = _make_cli()
        cli._agent_running = True
        assert (
            cli._should_handle_goal_command_inline(
                "/goal gate add sh -c 'pytest tests/'"
            )
            is True
        )
        assert cli._should_handle_goal_command_inline("/goal gate list") is True
        assert cli._should_handle_goal_command_inline("/goal gate") is True
        assert cli._should_handle_goal_command_inline("/goal gate remove 1") is True
        assert cli._should_handle_goal_command_inline("/goal gate clear") is True

    def test_detects_goal_control_commands_when_agent_running(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_goal_command_inline("/goal wait 1234") is True
        assert cli._should_handle_goal_command_inline("/goal unwait") is True
        assert cli._should_handle_goal_command_inline("/goal pause") is True
        assert cli._should_handle_goal_command_inline("/goal resume") is True
        assert cli._should_handle_goal_command_inline("/goal show") is True
        assert cli._should_handle_goal_command_inline("/goal status") is True
        assert cli._should_handle_goal_command_inline("/goal") is True

    def test_detects_subgoal_command_when_agent_running(self):
        cli = _make_cli()
        cli._agent_running = True
        assert (
            cli._should_handle_goal_command_inline("/subgoal all tests must pass")
            is True
        )

    def test_does_not_inline_new_goal_prompts_while_busy(self):
        """Setting a brand-new goal text is a new goal loop and queues normally."""
        cli = _make_cli()
        cli._agent_running = True
        assert (
            cli._should_handle_goal_command_inline(
                "/goal Create four files in notes/m04/"
            )
            is False
        )
        assert (
            cli._should_handle_goal_command_inline(
                "/goal draft Refactor payment gateway"
            )
            is False
        )

    def test_ignores_when_agent_idle(self):
        """Idle input falls through to the normal command dispatch."""
        cli = _make_cli()
        cli._agent_running = False
        assert (
            cli._should_handle_goal_command_inline(
                "/goal gate add sh -c 'pytest tests/'"
            )
            is False
        )

    def test_ignores_with_attached_images(self):
        cli = _make_cli()
        cli._agent_running = True
        assert (
            cli._should_handle_goal_command_inline(
                "/goal gate add test", has_images=True
            )
            is False
        )

    def test_ignores_non_slash_input(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_goal_command_inline("goal gate add test") is False
        assert cli._should_handle_goal_command_inline("") is False
