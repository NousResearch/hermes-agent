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


