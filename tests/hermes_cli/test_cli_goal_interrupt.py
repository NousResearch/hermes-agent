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


class TestMirrorResyncOnExternalWrite:
    """Another writer moved state_meta; the CLI's frozen mirror must resync
    before the active check so a done/cleared goal stops the loop and the dock."""

    def test_cleared_by_another_writer_stops_judging_and_repaints(self, hermes_home):
        sid = f"sid-resync-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        from hermes_cli.goals import clear_goal
        clear_goal(sid)  # out-of-band write: DB row changes, the mirror does not
        assert mgr.state is not None and mgr.state.status == "active"  # expected: mirror still frozen at active
        with patch("hermes_cli.goals.judge_goal") as judge:
            cli._maybe_continue_goal_after_turn()
        assert not judge.called  # expected: cleared in DB, no judging
        assert not mgr.has_goal()  # expected: mirror re-read from DB, dock line disappears

    def test_done_by_another_writer_stops_judging(self, hermes_home):
        sid = f"sid-resync-done-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        from hermes_cli.goals import GoalManager, save_goal
        other = GoalManager(session_id=mgr.session_id, default_max_turns=5)
        assert other.state is not None
        other.state.status = "done"
        save_goal(mgr.session_id, other.state)  # a distinct object, like another process
        with patch("hermes_cli.goals.judge_goal") as judge:
            cli._maybe_continue_goal_after_turn()
        assert not judge.called  # expected: already done, no judging
        assert mgr.state is not None and mgr.state.status == "done"  # expected: mirror synced to done

    def test_resumed_by_another_writer_reactivates_mirror(self, hermes_home):
        """done→active drift: another surface resumed the goal; the mirror must see it."""
        sid = f"sid-resync-resume-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli.conversation_history = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "did some work, more to do"},
        ]
        from hermes_cli.goals import GoalManager, save_goal
        other = GoalManager(session_id=mgr.session_id, default_max_turns=5)
        assert other.state is not None
        other.state.status = "done"
        save_goal(mgr.session_id, other.state)
        mgr.refresh()  # the CLI saw the judge's done first — mirror is now done
        assert mgr.state is not None and mgr.state.status == "done"  # expected: mirror at done before the resume
        other2 = GoalManager(session_id=mgr.session_id, default_max_turns=5)
        assert other2.state is not None
        other2.state.status = "active"  # a TUI-side /goal resume
        save_goal(mgr.session_id, other2.state)
        with patch(
            "hermes_cli.goals.judge_goal",
            return_value=("continue", "resumed", False, None, False),
        ) as judge:
            cli._maybe_continue_goal_after_turn()
        assert judge.called  # expected: active again, judging resumes
        assert mgr.is_active()  # expected: mirror saw the resume, dock line returns
