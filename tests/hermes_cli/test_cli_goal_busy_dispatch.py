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

import queue

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_cli():
    """Bare CLI, matching the upstream busy-command detector fixtures."""
    from cli import HermesCLI

    return HermesCLI.__new__(HermesCLI)


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

    @pytest.mark.parametrize("argument", [
        "continue", "unpause", "resume last goal", "pause the rollout", "clear", "stop", "done",
    ])
    def test_uses_shared_control_semantics(self, argument):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_goal_command_inline(f"/GOAL\t{argument}") is True

    @pytest.mark.parametrize("argument", ["-- pause the rollout", "clear the backlog", "stop the cron"])
    def test_literal_goal_text_stays_queued(self, argument):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_goal_command_inline(f"/goal {argument}") is False

    def test_inline_dispatch_attaches_gate_to_active_goal(self, tmp_path, monkeypatch):
        from hermes_cli import goals

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cli = _make_cli()
        cli._agent_running = True
        cli.agent = MagicMock()
        cli._pending_input = queue.Queue()
        manager = goals.GoalManager("busy-inline-goal")
        manager.set("keep working")
        cli._get_goal_manager = lambda: manager
        cli.process_command = cli._handle_goal_command
        event = SimpleNamespace(app=MagicMock())

        assert cli._tui_enter_inline_command(event, "/goal gate add pytest -q", False) is True

        assert manager.is_active()
        state = manager.state
        assert state is not None
        assert [gate.command for gate in state.gates] == ["pytest -q"]
        restored = goals.GoalManager("busy-inline-goal")
        restored_state = restored.state
        assert restored_state is not None
        assert [gate.command for gate in restored_state.gates] == ["pytest -q"]
        assert cli._pending_input.empty()
        assert cli.agent.method_calls == []
        event.app.current_buffer.reset.assert_called_once_with(append_to_history=True)
        event.app.invalidate.assert_called_once_with()

    def test_goal_controls_do_not_touch_running_agent(self):
        """Inline goal controls mutate GoalState only, never the active agent."""
        cli = _make_cli()
        cli._agent_running = True
        cli.agent = MagicMock()
        manager = MagicMock()
        manager.add_gate.return_value = SimpleNamespace(
            command="pytest -q",
            max_retries=1,
            timeout_seconds=30,
        )
        manager.has_goal.return_value = True
        manager.add_subgoal.return_value = "keep the API stable"
        manager.state = SimpleNamespace(subgoals=["keep the API stable"])
        cli._get_goal_manager = lambda: manager

        with patch("cli._cprint"):
            cli._handle_goal_command("/goal gate add pytest -q")
            cli._handle_subgoal_command("/subgoal keep the API stable")

        manager.add_gate.assert_called_once_with("pytest -q")
        manager.add_subgoal.assert_called_once_with("keep the API stable")
        assert cli.agent.method_calls == []
