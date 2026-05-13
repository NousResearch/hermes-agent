"""Tests for /loop CLI command — LoopManager-based same-session continuation.

Verifies that the CLI handler correctly dispatches subcommands,
parses interval prefixes, and integrates with LoopManager.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_cli():
    """Build a minimal HermesCLI stub with loop handler support."""
    from cli import HermesCLI
    cli = object.__new__(HermesCLI)
    cli._busy_command = lambda self, ctx: ctx
    cli._slow_command_status = lambda self, cmd: None
    cli._pending_input = MagicMock()
    cli._pending_input.empty.return_value = True
    cli.conversation_history = []
    cli._last_turn_interrupted = False
    # Patch _cprint to plain print() so capsys captures output and
    # prompt_toolkit does not try to flush to a closed stdout.
    import cli as _cli_mod
    _cli_mod._cprint = print
    return cli


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import loop

    loop._DB_CACHE.clear()
    yield home
    loop._DB_CACHE.clear()


# ------------------------------------------------------------------
# CLI /loop handler tests
# ------------------------------------------------------------------

class TestLoopCommandCLI:

    def test_dispatch_calls_handler(self):
        """process_command dispatches /loop to _handle_loop_command."""
        from hermes_cli.commands import resolve_command
        cmd_def = resolve_command("/loop")
        assert cmd_def is not None
        assert cmd_def.name == "loop"

    def test_no_args_shows_status_when_none(self, capsys, hermes_home):
        """/loop with no args shows 'No active loop' when none set."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "test-sid-1"

        cli._handle_loop_command("/loop")

        captured = capsys.readouterr()
        assert "No active loop" in captured.out

    def test_set_prompt(self, capsys, hermes_home):
        """/loop <prompt> creates a loop and kicks off immediately."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "test-sid-2"

        cli._handle_loop_command("/loop check deployment")

        captured = capsys.readouterr()
        assert "Loop set" in captured.out
        assert "check deployment" in captured.out
        assert "300s interval" in captured.out  # default
        cli._pending_input.put.assert_called_once_with("check deployment")

    def test_set_with_interval(self, capsys, hermes_home):
        """/loop 5m check deployment parses interval prefix."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "test-sid-3"

        cli._handle_loop_command("/loop 5m check deployment")

        captured = capsys.readouterr()
        assert "Loop set" in captured.out
        assert "300s interval" in captured.out
        assert "check deployment" in captured.out
        cli._pending_input.put.assert_called_once_with("check deployment")

    def test_set_every_syntax(self, capsys, hermes_home):
        """/loop every 30m summarize news parses 'every' prefix."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "test-sid-4"

        cli._handle_loop_command("/loop every 30m summarize news")

        captured = capsys.readouterr()
        assert "Loop set" in captured.out
        assert "1800s interval" in captured.out
        assert "summarize news" in captured.out

    def test_set_empty_prompt_error(self, capsys, hermes_home):
        """/loop 5m with no prompt shows usage."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "test-sid-5"

        cli._handle_loop_command("/loop 5m")

        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_status_shows_active_loop(self, capsys, hermes_home):
        """/loop status shows active loop details."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "test-sid-6"
        cli._handle_loop_command("/loop check deployment")

        cli._pending_input.reset_mock()
        cli._handle_loop_command("/loop status")

        captured = capsys.readouterr()
        assert "active" in captured.out.lower()
        assert "check deployment" in captured.out

    def test_pause_subcommand(self, capsys, hermes_home):
        """/loop pause pauses the active loop."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "test-sid-7"
        cli._handle_loop_command("/loop check deployment")

        cli._pending_input.reset_mock()
        cli._handle_loop_command("/loop pause")

        captured = capsys.readouterr()
        assert "Loop paused" in captured.out
        assert "check deployment" in captured.out

    def test_resume_subcommand(self, capsys, hermes_home):
        """/loop resume resumes a paused loop."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "test-sid-8"
        cli._handle_loop_command("/loop check deployment")
        cli._handle_loop_command("/loop pause")

        cli._pending_input.reset_mock()
        cli._handle_loop_command("/loop resume")

        captured = capsys.readouterr()
        assert "Loop resumed" in captured.out
        assert "check deployment" in captured.out

    def test_clear_subcommand(self, capsys, hermes_home):
        """/loop clear clears the active loop."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "test-sid-9"
        cli._handle_loop_command("/loop check deployment")

        cli._pending_input.reset_mock()
        cli._handle_loop_command("/loop clear")

        captured = capsys.readouterr()
        assert "Loop cleared" in captured.out

    def test_clear_when_no_loop(self, capsys, hermes_home):
        """/loop clear when no loop shows friendly message."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "test-sid-10"

        cli._handle_loop_command("/loop clear")

        captured = capsys.readouterr()
        assert "No active loop" in captured.out

    def test_pause_when_no_loop(self, capsys, hermes_home):
        """/loop pause when no loop shows friendly message."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "test-sid-11"

        cli._handle_loop_command("/loop pause")

        captured = capsys.readouterr()
        assert "No loop set" in captured.out

    def test_resume_when_no_loop(self, capsys, hermes_home):
        """/loop resume when no loop shows friendly message."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "test-sid-12"

        cli._handle_loop_command("/loop resume")

        captured = capsys.readouterr()
        assert "No loop to resume" in captured.out


# ------------------------------------------------------------------
# _maybe_continue_loop_after_turn hook tests
# ------------------------------------------------------------------
class TestLoopContinuationHook:
    """_maybe_continue_loop_after_turn is now a no-op — loop is driven by
    the LoopScheduler background daemon thread, not a post-turn hook."""

    def test_no_op_does_not_error(self, hermes_home):
        """No-op hook doesn't raise on any state."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "hook-sid-1"
        cli._maybe_continue_loop_after_turn()

    def test_no_op_never_enqueues(self, hermes_home):
        """No-op hook never calls _pending_input.put."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "hook-sid-2"

        from hermes_cli.loop import LoopManager
        loop_mgr = LoopManager(session_id="hook-sid-2")
        loop_mgr.set("check deployment")
        cli._loop_manager = loop_mgr
        cli.conversation_history = [{"role": "assistant", "content": "all good"}]

        cli._maybe_continue_loop_after_turn()
        cli._pending_input.put.assert_not_called()

    def test_no_op_never_pauses(self, hermes_home):
        """No-op hook does not pause loop on interrupt."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "hook-sid-3"
        cli._last_turn_interrupted = True

        from hermes_cli.loop import LoopManager
        loop_mgr = LoopManager(session_id="hook-sid-3")
        loop_mgr.set("check deployment")
        cli._loop_manager = loop_mgr

        cli._maybe_continue_loop_after_turn()
        assert loop_mgr.state.status == "active"

    def test_no_op_with_goal_active(self, hermes_home):
        """No-op hook doesn't error even with active goal."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "hook-sid-4"

        from hermes_cli.goals import GoalManager
        goal_mgr = GoalManager(session_id="hook-sid-4")
        goal_mgr.set("do the thing")
        cli._goal_manager = goal_mgr

        from hermes_cli.loop import LoopManager
        loop_mgr = LoopManager(session_id="hook-sid-4")
        loop_mgr.set("check deployment")
        cli._loop_manager = loop_mgr

        cli._maybe_continue_loop_after_turn()
