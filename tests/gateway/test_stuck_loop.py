"""Tests for stuck-session loop detection (#7536).

When a session is active across 3+ consecutive gateway restarts (the agent
gets stuck, gateway restarts, same session gets stuck again), the session
is auto-suspended on startup so the user gets a clean slate.
"""

import json
from unittest.mock import MagicMock

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.fixture
def runner_with_home(tmp_path, monkeypatch):
    """Create a runner with a writable HERMES_HOME."""
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    return runner, tmp_path


class TestStuckLoopDetection:

    def test_increment_creates_file(self, runner_with_home):
        runner, home = runner_with_home
        runner._increment_restart_failure_counts({"session:a", "session:b"})
        path = home / runner._STUCK_LOOP_FILE
        assert path.exists()
        counts = json.loads(path.read_text())
        assert counts["session:a"] == 1
        assert counts["session:b"] == 1

    def test_increment_accumulates(self, runner_with_home):
        runner, home = runner_with_home
        runner._increment_restart_failure_counts({"session:a"})
        runner._increment_restart_failure_counts({"session:a"})
        runner._increment_restart_failure_counts({"session:a"})
        counts = json.loads((home / runner._STUCK_LOOP_FILE).read_text())
        assert counts["session:a"] == 3

    def test_increment_drops_inactive_sessions(self, runner_with_home):
        runner, home = runner_with_home
        runner._increment_restart_failure_counts({"session:a", "session:b"})
        runner._increment_restart_failure_counts({"session:a"})  # b not active
        counts = json.loads((home / runner._STUCK_LOOP_FILE).read_text())
        assert "session:a" in counts
        assert "session:b" not in counts

    def test_suspend_at_threshold(self, runner_with_home):
        runner, home = runner_with_home
        # Simulate 3 restarts with session:a active each time
        for _ in range(3):
            runner._increment_restart_failure_counts({"session:a"})

        # Create a mock session entry
        mock_entry = MagicMock()
        mock_entry.suspended = False
        runner.session_store._entries = {"session:a": mock_entry}
        runner.session_store._save = MagicMock()

        suspended = runner._suspend_stuck_loop_sessions()
        assert suspended == 1
        assert mock_entry.suspended is True

    def test_no_suspend_below_threshold(self, runner_with_home):
        runner, home = runner_with_home
        runner._increment_restart_failure_counts({"session:a"})
        runner._increment_restart_failure_counts({"session:a"})
        # Only 2 restarts — below threshold of 3

        mock_entry = MagicMock()
        mock_entry.suspended = False
        runner.session_store._entries = {"session:a": mock_entry}

        suspended = runner._suspend_stuck_loop_sessions()
        assert suspended == 0
        assert mock_entry.suspended is False

    def test_clear_on_success(self, runner_with_home):
        runner, home = runner_with_home
        runner._increment_restart_failure_counts({"session:a", "session:b"})
        runner._clear_restart_failure_count("session:a")

        path = home / runner._STUCK_LOOP_FILE
        counts = json.loads(path.read_text())
        assert "session:a" not in counts
        assert "session:b" in counts

    def test_clear_removes_file_when_empty(self, runner_with_home):
        runner, home = runner_with_home
        runner._increment_restart_failure_counts({"session:a"})
        runner._clear_restart_failure_count("session:a")
        assert not (home / runner._STUCK_LOOP_FILE).exists()

    def test_suspend_clears_file(self, runner_with_home):
        runner, home = runner_with_home
        for _ in range(3):
            runner._increment_restart_failure_counts({"session:a"})

        mock_entry = MagicMock()
        mock_entry.suspended = False
        runner.session_store._entries = {"session:a": mock_entry}
        runner.session_store._save = MagicMock()

        runner._suspend_stuck_loop_sessions()
        assert not (home / runner._STUCK_LOOP_FILE).exists()

    def test_no_file_no_crash(self, runner_with_home):
        runner, home = runner_with_home
        # No file exists — should return 0 and not crash
        assert runner._suspend_stuck_loop_sessions() == 0
        # Clear on nonexistent file — should not crash
        runner._clear_restart_failure_count("nonexistent")


class TestResetStuckLoopCounts:
    """The set-based reset used by the clean-drain path (#7536 false-suspend fix).

    Distinct from the singular ``_clear_restart_failure_count`` (drops the whole
    entry on a successful turn): this resets the COUNT axis for a set of
    drained-clean sessions while preserving the separate replay-loop breaker's
    state (``replay_marks``/``armed``).
    """

    def test_reset_clears_plain_count(self, runner_with_home):
        runner, home = runner_with_home
        runner._increment_restart_failure_counts({"session:a", "session:b"})
        runner._increment_restart_failure_counts({"session:a", "session:b"})
        # session:a drained cleanly this restart → its count resets to 0.
        runner._reset_stuck_loop_counts({"session:a"})
        counts = json.loads((home / runner._STUCK_LOOP_FILE).read_text())
        assert "session:a" not in counts  # pruned (no replay state, count 0)
        assert counts["session:b"] == 2  # untouched

    def test_reset_preserves_replay_state(self, runner_with_home):
        runner, home = runner_with_home
        # Seed an entry that also carries replay-loop breaker state.
        path = home / runner._STUCK_LOOP_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "session:a": {"count": 2, "replay_marks": [123.0], "armed": True},
        }))
        runner._reset_stuck_loop_counts({"session:a"})
        counts = json.loads(path.read_text())
        # Count reset, but replay-breaker state (the SEPARATE mechanism) kept.
        assert counts["session:a"]["count"] == 0
        assert counts["session:a"]["replay_marks"] == [123.0]
        assert counts["session:a"]["armed"] is True

    def test_reset_empty_set_is_noop(self, runner_with_home):
        runner, home = runner_with_home
        runner._increment_restart_failure_counts({"session:a"})
        runner._reset_stuck_loop_counts(set())
        counts = json.loads((home / runner._STUCK_LOOP_FILE).read_text())
        assert counts["session:a"] == 1

    def test_reset_missing_key_no_crash(self, runner_with_home):
        runner, home = runner_with_home
        # No file / unknown key — must not crash.
        runner._reset_stuck_loop_counts({"nonexistent"})
        runner._increment_restart_failure_counts({"session:a"})
        runner._reset_stuck_loop_counts({"nonexistent"})
        counts = json.loads((home / runner._STUCK_LOOP_FILE).read_text())
        assert counts["session:a"] == 1


class TestSuccessfulTurnResetsStuckLoopCount:
    """2026-07-10 live incident: a healthy long-running session was busy at
    3 consecutive gateway restarts (other sessions deploying), each drain
    timed out, the counter marched 1→2→3 despite the session completing
    real turns between restarts, and _suspend_stuck_loop_sessions falsely
    suspended it — surfacing as "Session automatically reset ... history
    cleared".

    Contract: a successful turn is affirmative proof the session is NOT
    stuck; the post-turn gate (_apply_post_turn_resume_gate, non-initiator
    branch) must drop the session's stuck-loop entry, restoring the
    original _clear_restart_failure_count contract whose call site was
    lost in the F2-breaker refactor.
    """

    def _gate_runner(self, runner):
        """Wire the minimal collaborators the post-turn gate touches."""
        runner.session_store = MagicMock()
        runner._consume_restart_initiated_breadcrumb = lambda _sk: False
        return runner

    def test_incident_sequence_no_false_suspend(self, runner_with_home):
        """Interrupt, interrupt, SUCCESSFUL TURN, interrupt — must NOT
        suspend (pre-fix: count reached 3 because success didn't reset)."""
        runner, home = runner_with_home
        self._gate_runner(runner)
        sk = "agent:main:discord:thread:kanban:kanban"

        runner._increment_restart_failure_counts({sk})   # restart 1
        runner._increment_restart_failure_counts({sk})   # restart 2

        # A real turn completes between restarts (the kanban session did
        # this repeatedly — 653s turns finishing fine).
        runner._apply_post_turn_resume_gate(sk)

        runner._increment_restart_failure_counts({sk})   # restart 3

        mock_entry = MagicMock()
        mock_entry.suspended = False
        runner.session_store._entries = {sk: mock_entry}
        runner.session_store._save = MagicMock()

        suspended = runner._suspend_stuck_loop_sessions()
        assert suspended == 0, (
            "healthy session falsely suspended: successful turn did not "
            "reset the stuck-loop count"
        )
        assert mock_entry.suspended is False

    def test_gate_clears_count_for_non_initiator(self, runner_with_home):
        runner, home = runner_with_home
        self._gate_runner(runner)
        sk = "session:worked"
        runner._increment_restart_failure_counts({sk})
        runner._increment_restart_failure_counts({sk})

        runner._apply_post_turn_resume_gate(sk)

        path = home / runner._STUCK_LOOP_FILE
        if path.exists():
            counts = json.loads(path.read_text())
            assert sk not in counts
        # else: file pruned entirely — also correct

    def test_gate_keeps_count_for_restart_initiator(self, runner_with_home):
        """A turn whose only outcome was ANOTHER restart is loop progress,
        not work progress — the count must survive (F2 contract)."""
        runner, home = runner_with_home
        self._gate_runner(runner)
        sk = "session:restarter"
        runner._increment_restart_failure_counts({sk})
        runner._session_initiated_restart[sk] = True

        runner._apply_post_turn_resume_gate(sk)

        counts = json.loads((home / runner._STUCK_LOOP_FILE).read_text())
        assert sk in counts
        assert counts[sk]["count"] == 1

    def test_genuinely_stuck_session_still_suspended(self, runner_with_home):
        """The detector's purpose survives: 3 interruptions with NO
        successful turn between them still suspends."""
        runner, home = runner_with_home
        self._gate_runner(runner)
        sk = "session:stuck"
        for _ in range(3):
            runner._increment_restart_failure_counts({sk})

        mock_entry = MagicMock()
        mock_entry.suspended = False
        runner.session_store._entries = {sk: mock_entry}
        runner.session_store._save = MagicMock()

        assert runner._suspend_stuck_loop_sessions() == 1
        assert mock_entry.suspended is True
