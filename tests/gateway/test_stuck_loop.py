"""Tests for stuck-session loop detection (#7536).

When a session is active across 3+ consecutive gateway restarts (the agent
gets stuck, gateway restarts, same session gets stuck again), the session
is auto-suspended on startup so the user gets a clean slate.
"""

import json
from pathlib import Path
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
