"""Tests for stuck-session loop detection (#7536).

When a session is active across 3+ consecutive gateway restarts (the agent
gets stuck, gateway restarts, same session gets stuck again), the session
is auto-suspended on startup so the user gets a clean slate.
"""

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner
from tests.lock_contention import ContentionObservedLock
from tests.waiting import wait_until


@pytest.fixture
def runner_with_home(tmp_path, monkeypatch):
    """Create a runner with a writable HERMES_HOME."""
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner, _adapter = make_restart_runner()
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
        runner, _home = runner_with_home
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
        runner, _home = runner_with_home
        runner._increment_restart_failure_counts({"session:a"})
        runner._increment_restart_failure_counts({"session:a"})
        # Only 2 restarts — below threshold of 3

        mock_entry = MagicMock()
        mock_entry.suspended = False
        runner.session_store._entries = {"session:a": mock_entry}

        suspended = runner._suspend_stuck_loop_sessions()
        assert suspended == 0
        assert mock_entry.suspended is False

    def test_concurrent_clears_do_not_resurrect_each_others_counter(self, runner_with_home, monkeypatch):
        """Two sessions completing at once each clear their counter; neither may rewrite the other's
        stale snapshot (race audit lostupdate-6: a resurrected counter can later auto-suspend a healthy session)."""
        import gateway.stuck_loop_counts as counts_module
        import utils
        runner, home = runner_with_home
        path = home / runner._STUCK_LOOP_FILE
        counts_lock = ContentionObservedLock()
        monkeypatch.setattr(counts_module, "_COUNTS_LOCK", counts_lock)
        path.write_text(json.dumps({"session:a": 2, "session:b": 2}))
        reads = []
        second_read = threading.Event()
        real_read_text = Path.read_text

        def counting_read_text(self, *args, **kwargs):
            if self == path:
                reads.append(self)
                if len(reads) == 2:
                    second_read.set()
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", counting_read_text)
        real_write = utils.atomic_json_write
        first_write = threading.Event()

        def parked_write(target, data, **kwargs):
            if not first_write.is_set():
                first_write.set()
                # Unserialized clears let the other session read the not-yet-rewritten file here;
                # serialized ones are seen blocking on the counters lock instead.
                wait_until(lambda: second_read.is_set() or counts_lock.contended.is_set(),
                           what="the other clear read the file or blocked on the counters lock")
            return real_write(target, data, **kwargs)

        monkeypatch.setattr(utils, "atomic_json_write", parked_write)

        async def complete_both():
            await asyncio.gather(
                runner._clear_restart_failure_count("session:a"),
                runner._clear_restart_failure_count("session:b"),
            )

        asyncio.run(complete_both())
        remaining = json.loads(real_read_text(path)) if path.exists() else {}
        assert remaining == {}
