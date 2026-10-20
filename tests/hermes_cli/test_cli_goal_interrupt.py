"""Tests for CLI goal-continuation interrupt handling.

Covers:
- Ctrl+C during a /goal turn auto-pauses the goal (no more continuations).
- Empty/whitespace-only responses skip the judge (no phantom continuations).
- Clean response without interrupt still drives the judge + enqueues.

These tests exercise ``_maybe_continue_goal_after_turn`` directly on a
minimal ``HermesCLI`` stub (pattern used elsewhere in tests/cli).
"""

from __future__ import annotations

import queue
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes stay hermetic."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal module's DB cache so it re-resolves HERMES_HOME each test.
    from hermes_cli import goals
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()

def _make_cli_with_goal(session_id: str, goal_text: str = "build a thing"):
    """Build a minimal HermesCLI stub with an active goal wired in."""
    from cli import HermesCLI
    from hermes_cli.goals import GoalManager

    cli = HermesCLI.__new__(HermesCLI)
    # State the hook + helpers touch directly.
    cli._pending_input = queue.Queue()
    cli._last_turn_interrupted = False
    cli.conversation_history = []
    # `_get_goal_manager()` reads `self.session_id` directly, not
    # `self.agent.session_id`. Match the production lookup.
    cli.session_id = session_id
    cli.agent = MagicMock()
    cli.agent.session_id = session_id

    mgr = GoalManager(session_id=session_id, default_max_turns=5)
    mgr.set(goal_text)
    cli._goal_manager = mgr
    return cli, mgr

# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

class TestInterruptAutoPause:

    def test_interrupted_turn_is_resumable(self, hermes_home):
        """After auto-pause from Ctrl+C, /goal resume puts it back to active."""
        sid = f"sid-resume-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = True
        cli.conversation_history = [
            {"role": "assistant", "content": "partial"},
        ]
        with patch("hermes_cli.goals.judge_goal"):
            cli._maybe_continue_goal_after_turn()
        assert mgr.state.status == "paused"

        mgr.resume()
        assert mgr.state.status == "active"

class TestHealthyTurnStillRuns:
    def test_clean_response_enqueues_continuation_when_judge_says_continue(
        self, hermes_home,
    ):
        """Sanity check: the hook still works in the happy path."""
        sid = f"sid-healthy-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = False
        cli.conversation_history = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "did some work, more to do"},
        ]

        # Force the judge to say "continue" without touching the network.
        with patch(
            "hermes_cli.goals.judge_goal",
            return_value=("continue", "needs more steps", False, None, False),
        ):
            cli._maybe_continue_goal_after_turn()

        # Continuation prompt must be queued.
        assert not cli._pending_input.empty()
        queued = cli._pending_input.get_nowait()
        assert "Continuing toward your standing goal" in queued
        assert mgr.state.status == "active"

    def test_clean_response_marks_done_when_judge_says_done(self, hermes_home):
        sid = f"sid-done-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = False
        cli.conversation_history = [
            {"role": "assistant", "content": "all finished, here's the result"},
        ]

        with patch(
            "hermes_cli.goals.judge_goal",
            return_value=("done", "goal satisfied", False, None, False),
        ):
            cli._maybe_continue_goal_after_turn()

        assert cli._pending_input.empty()
        assert mgr.state.status == "done"

_JUDGE_CONTINUE = ("continue", "needs more steps", False, None, False)

class TestFailedTurnSkipsJudge:
    """A failed provider turn has no model reply: the newest assistant row is the failed-turn
    notice (or an earlier turn's reply). Judging it re-queued continuations into the same failure
    until the turn budget ran out."""

    def _cli_after_failed_turn(self, result):
        from agent.turn_failure_copy import FAILED_TURN_NOTICE
        cli, mgr = _make_cli_with_goal(f"sid-failed-{uuid.uuid4().hex}")
        cli.conversation_history = [
            {"role": "user", "content": "do cycle 3"},
            {"role": "assistant", "content": "Cycle 3 done; validate PASS. Next: cycle 4."},
            {"role": "user", "content": "[Continuing toward your standing goal]"},
            {"role": "assistant", "content": FAILED_TURN_NOTICE},
        ]
        cli._last_turn_result = result
        return cli, mgr

    @pytest.mark.parametrize("reason", ["billing", "auth", "auth_permanent"])
    def test_billing_or_auth_failure_pauses_without_judging(self, hermes_home, reason):
        cli, mgr = self._cli_after_failed_turn(
            {"failed": True, "completed": False, "failure_reason": reason, "final_response": None})
        with patch("hermes_cli.goals.judge_goal", return_value=_JUDGE_CONTINUE) as judge:
            cli._maybe_continue_goal_after_turn()
        judge.assert_not_called()
        assert cli._pending_input.empty()
        assert mgr.state.status == "paused"
        assert reason in mgr.state.paused_reason

    def test_transient_failure_skips_judge_and_keeps_goal_active(self, hermes_home):
        cli, mgr = self._cli_after_failed_turn(
            {"failed": True, "completed": False, "failure_reason": "server_error"})
        with patch("hermes_cli.goals.judge_goal", return_value=_JUDGE_CONTINUE) as judge:
            cli._maybe_continue_goal_after_turn()
        judge.assert_not_called()
        assert cli._pending_input.empty()
        assert mgr.state.status == "active"
        assert mgr.state.turns_used == 0

    def test_max_iterations_handoff_still_reaches_the_judge(self, hermes_home):
        """#102213: a non-failed max-iterations handoff is a turn boundary, not a failure."""
        cli, mgr = self._cli_after_failed_turn({
            "failed": False, "interrupted": False, "completed": False,
            "turn_exit_reason": "max_iterations_reached(90/90)",
            "final_response": "Summary of progress so far."})
        cli.conversation_history.append({"role": "assistant", "content": "Summary of progress so far."})
        with patch("hermes_cli.goals.judge_goal", return_value=_JUDGE_CONTINUE) as judge:
            cli._maybe_continue_goal_after_turn()
        judge.assert_called_once()
        assert not cli._pending_input.empty()
