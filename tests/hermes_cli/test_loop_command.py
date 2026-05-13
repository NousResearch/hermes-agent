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

    def test_goal_takes_priority(self, hermes_home):
        """When goal is active, loop hook skips."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "hook-sid-1"

        # Set a goal (active)
        from hermes_cli.goals import GoalManager
        goal_mgr = GoalManager(session_id="hook-sid-1")
        goal_mgr.set("do the thing")
        cli._goal_manager = goal_mgr

        # Set a loop (also active)
        from hermes_cli.loop import LoopManager
        loop_mgr = LoopManager(session_id="hook-sid-1")
        loop_mgr.set("check deployment")
        cli._loop_manager = loop_mgr

        # Goal takes priority — loop should not enqueue
        cli._maybe_continue_loop_after_turn()
        cli._pending_input.put.assert_not_called()

    def test_loop_continues_when_interval_elapsed(self, hermes_home):
        """Loop hook enqueues continuation when interval elapsed."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "hook-sid-2"

        from hermes_cli.loop import LoopManager
        loop_mgr = LoopManager(session_id="hook-sid-2")
        loop_mgr.set("check deployment")
        cli._loop_manager = loop_mgr

        # Give it a non-empty last response
        cli.conversation_history = [{"role": "assistant", "content": "all good"}]

        cli._maybe_continue_loop_after_turn()
        # First call: interval always passes (last_fired_at=0)
        cli._pending_input.put.assert_called_once()
        args, _ = cli._pending_input.put.call_args
        assert "[Loop check] check deployment" in args[0]

    def test_loop_skips_when_interval_not_elapsed(self, hermes_home):
        """Loop hook skips when interval hasn't elapsed."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "hook-sid-3"

        from hermes_cli.loop import LoopManager
        loop_mgr = LoopManager(session_id="hook-sid-3")
        loop_mgr.set("check deployment", interval_seconds=300)
        loop_mgr.state.last_fired_at = __import__("time").time()
        cli._loop_manager = loop_mgr

        cli.conversation_history = [{"role": "assistant", "content": "all good"}]

        cli._maybe_continue_loop_after_turn()
        cli._pending_input.put.assert_not_called()

    def test_loop_skips_on_empty_response(self, hermes_home):
        """Loop hook skips when last response is empty."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "hook-sid-4"

        from hermes_cli.loop import LoopManager
        loop_mgr = LoopManager(session_id="hook-sid-4")
        loop_mgr.set("check deployment")
        cli._loop_manager = loop_mgr

        cli.conversation_history = [{"role": "assistant", "content": ""}]

        cli._maybe_continue_loop_after_turn()
        cli._pending_input.put.assert_not_called()

    def test_loop_skips_on_interrupt(self, hermes_home):
        """Loop hook pauses loop on interrupt."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "hook-sid-5"
        cli._last_turn_interrupted = True

        from hermes_cli.loop import LoopManager
        loop_mgr = LoopManager(session_id="hook-sid-5")
        loop_mgr.set("check deployment")
        cli._loop_manager = loop_mgr

        cli._maybe_continue_loop_after_turn()
        cli._pending_input.put.assert_not_called()
        assert loop_mgr.state.status == "paused"

    def test_loop_skips_when_user_message_queued(self, hermes_home):
        """Loop hook skips when real user message is already queued."""
        from cli import HermesCLI
        cli = _make_cli()
        cli.session_id = "hook-sid-6"
        cli._pending_input.empty.return_value = False

        from hermes_cli.loop import LoopManager
        loop_mgr = LoopManager(session_id="hook-sid-6")
        loop_mgr.set("check deployment")
        cli._loop_manager = loop_mgr

        cli._maybe_continue_loop_after_turn()
        cli._pending_input.put.assert_not_called()
