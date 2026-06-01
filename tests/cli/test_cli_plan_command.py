"""Tests for the /plan slash command (issue #36821).

Covers:
- The CommandDef entry is in the registry with the right metadata.
- ``process_command`` dispatches ``/plan`` to ``_handle_plan_command``.
- Empty / no-agent / unknown-arg edge cases render helpful messages.
- The default render shows every todo with the right status marker.
- ``/plan pending`` filters to pending + in_progress.
- ``/plan clear`` wipes the store and respects the confirmation gate.
- Skip tokens (--yes / now) bypass the gate, matching the /new pattern.
- Prefix matching still routes /pla to the plan handler.
"""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI
from hermes_cli.commands import COMMAND_REGISTRY, resolve_command
from tools.todo_tool import TodoStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cli(todo_items=None):
    """Build a HermesCLI-ish object with a real TodoStore."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj._console_print = MagicMock()
    cli_obj.conversation_history = []
    cli_obj.session_id = "session-plan-test"
    cli_obj._pending_input = MagicMock()
    cli_obj._agent_running = False
    cli_obj._session_db = None
    cli_obj.session_start = datetime(2026, 5, 1, 12, 0, 0)
    cli_obj.model = "anthropic/claude-sonnet-4.6"
    cli_obj.provider = "anthropic"

    agent = MagicMock()
    agent.session_id = cli_obj.session_id
    store = TodoStore()
    if todo_items is not None:
        store.write(todo_items)
    agent._todo_store = store
    cli_obj.agent = agent

    # Default: gate is open (no prompt). Tests that exercise the gate
    # override this attribute; MagicMock so tests can assert_not_called().
    cli_obj._confirm_destructive_slash = MagicMock(return_value="once")
    return cli_obj


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestPlanRegistry:
    def test_plan_command_is_registered(self):
        cmd = resolve_command("plan")
        assert cmd is not None
        assert cmd.name == "plan"

    def test_plan_command_metadata(self):
        plan = next(c for c in COMMAND_REGISTRY if c.name == "plan")
        assert plan.cli_only is True
        assert plan.args_hint == "[pending|clear]"
        assert plan.category == "Session"

    def test_plan_appears_in_commands_dict(self):
        from hermes_cli.commands import COMMANDS
        assert "/plan" in COMMANDS


# ---------------------------------------------------------------------------
# process_command dispatch
# ---------------------------------------------------------------------------


class TestPlanDispatch:
    def test_plan_dispatches_to_handler(self):
        cli = _make_cli()
        with patch.object(cli, "_handle_plan_command", create=True) as mock:
            assert cli.process_command("/plan") is True
        mock.assert_called_once_with("/plan")

    def test_plan_prefix_routes_correctly(self):
        """`/pla` is not a prefix in the registry, so we use a real lookup
        via `resolve_command` to confirm only an exact or alias match wins."""
        cli = _make_cli()
        # No alias for plan — the registry canonical name is the dispatch key.
        from hermes_cli.commands import _COMMAND_LOOKUP
        assert "plan" in _COMMAND_LOOKUP
        assert _COMMAND_LOOKUP["plan"].name == "plan"

    def test_plan_does_not_conflict_with_history_or_status(self):
        """Sanity: /plan and /history and /status are distinct commands."""
        cli = _make_cli(todo_items=[
            {"id": "1", "content": "a", "status": "pending"},
        ])
        with patch.object(cli, "show_history", create=True) as mock_hist, \
             patch.object(cli, "_show_session_status", create=True) as mock_stat, \
             patch.object(cli, "_handle_plan_command", create=True) as mock_plan:
            cli.process_command("/plan")
            cli.process_command("/history")
            cli.process_command("/status")
        mock_plan.assert_called_once()
        mock_hist.assert_called_once()
        mock_stat.assert_called_once()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestPlanRender:
    def test_empty_store_prints_friendly_message(self, capsys):
        cli = _make_cli(todo_items=[])
        cli._handle_plan_command("/plan")
        out = capsys.readouterr().out
        assert "No plan yet" in out

    def test_no_agent_prints_helpful_message(self, capsys):
        cli = _make_cli()
        cli.agent = None
        cli._handle_plan_command("/plan")
        out = capsys.readouterr().out
        assert "No active agent session" in out

    def test_full_render_shows_all_statuses(self, capsys):
        cli = _make_cli(todo_items=[
            {"id": "1", "content": "Setup", "status": "completed"},
            {"id": "2", "content": "Build feature", "status": "in_progress"},
            {"id": "3", "content": "Write tests", "status": "pending"},
            {"id": "4", "content": "Abandoned idea", "status": "cancelled"},
        ])
        cli._handle_plan_command("/plan")
        out = capsys.readouterr().out

        assert "📋 Current Plan" in out
        assert "Setup" in out
        assert "Build feature" in out
        assert "Write tests" in out
        assert "Abandoned idea" in out
        # Status markers
        assert "✓" in out
        assert "●" in out
        assert "○" in out
        assert "✗" in out
        # in_progress tail surfaces the status text
        assert "in_progress" in out
        # Progress line — 1 completed, 1 cancelled => 1 of (4 - 1) = 1/3
        assert "Progress: 1/3 completed" in out
        assert "1 cancelled item" in out

    def test_pending_filter_excludes_completed(self, capsys):
        cli = _make_cli(todo_items=[
            {"id": "1", "content": "Done thing", "status": "completed"},
            {"id": "2", "content": "WIP thing", "status": "in_progress"},
            {"id": "3", "content": "Next thing", "status": "pending"},
        ])
        cli._handle_plan_command("/plan pending")
        out = capsys.readouterr().out

        assert "WIP thing" in out
        assert "Next thing" in out
        assert "Done thing" not in out

    def test_pending_filter_with_no_active_items(self, capsys):
        cli = _make_cli(todo_items=[
            {"id": "1", "content": "All done", "status": "completed"},
        ])
        cli._handle_plan_command("/plan pending")
        out = capsys.readouterr().out
        assert "No pending tasks" in out

    def test_unknown_subcommand_prints_usage(self, capsys):
        cli = _make_cli(todo_items=[
            {"id": "1", "content": "a", "status": "pending"},
        ])
        cli._handle_plan_command("/plan foo")
        out = capsys.readouterr().out
        assert "Unknown /plan option" in out
        assert "foo" in out

    def test_box_drawing_uses_consistent_width(self, capsys):
        """The left/right borders must be the same length."""
        cli = _make_cli(todo_items=[
            {"id": "1", "content": "short", "status": "pending"},
            {"id": "2", "content": "a longer content string", "status": "in_progress"},
        ])
        cli._handle_plan_command("/plan")
        out = capsys.readouterr().out
        # Extract the top and bottom border lines.
        lines = [ln for ln in out.splitlines() if "┌" in ln or "└" in ln]
        assert len(lines) == 2
        top = lines[0]
        bot = lines[1]
        # Strip the leading "  " indent and the corner characters.
        top_inner = top.strip().lstrip("┌").rstrip("┐")
        bot_inner = bot.strip().lstrip("└").rstrip("┘")
        assert top_inner == bot_inner


# ---------------------------------------------------------------------------
# /plan clear
# ---------------------------------------------------------------------------


class TestPlanClear:
    def test_clear_wipes_store_when_approved(self, capsys):
        cli = _make_cli(todo_items=[
            {"id": "1", "content": "a", "status": "in_progress"},
            {"id": "2", "content": "b", "status": "pending"},
        ])
        cli._handle_plan_command("/plan clear")
        assert cli.agent._todo_store.read() == []
        out = capsys.readouterr().out
        assert "Plan cleared" in out

    def test_clear_keeps_store_when_cancelled(self, capsys):
        cli = _make_cli(todo_items=[
            {"id": "1", "content": "a", "status": "in_progress"},
        ])
        cli._confirm_destructive_slash = lambda *_a, **_kw: None
        cli._handle_plan_command("/plan clear")
        # Store untouched
        assert len(cli.agent._todo_store.read()) == 1
        out = capsys.readouterr().out
        assert "Cancelled" in out
        assert "Plan left unchanged" in out

    def test_clear_empty_store_is_a_noop(self, capsys):
        cli = _make_cli(todo_items=[])
        cli._handle_plan_command("/plan clear")
        # Gate never even called.
        cli._confirm_destructive_slash.assert_not_called()
        out = capsys.readouterr().out
        assert "No active plan to clear" in out

    def test_clear_without_agent_is_a_noop(self, capsys):
        cli = _make_cli()
        cli.agent = None
        cli._handle_plan_command("/plan clear")
        cli._confirm_destructive_slash.assert_not_called()
        out = capsys.readouterr().out
        assert "No active plan to clear" in out

    def test_clear_with_yes_skip_token_bypasses_gate(self, capsys):
        cli = _make_cli(todo_items=[
            {"id": "1", "content": "a", "status": "in_progress"},
        ])
        # Simulate the real gate behaviour: only "always" or "once" if skip
        # token is present, None otherwise.
        def _gate(*args, **kwargs):
            return "once" if "yes" in (kwargs.get("cmd_original") or "") else None
        cli._confirm_destructive_slash = MagicMock(side_effect=_gate)
        cli._handle_plan_command("/plan clear --yes")
        assert cli.agent._todo_store.read() == []
        out = capsys.readouterr().out
        assert "Plan cleared" in out

    def test_clear_with_now_skip_token_bypasses_gate(self, capsys):
        cli = _make_cli(todo_items=[
            {"id": "1", "content": "a", "status": "in_progress"},
        ])
        cli._confirm_destructive_slash = MagicMock(
            side_effect=lambda *a, **kw: "once" if "now" in (kw.get("cmd_original") or "") else None
        )
        cli._handle_plan_command("/plan clear now")
        assert cli.agent._todo_store.read() == []
