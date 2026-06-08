"""Tests for the kanban dispatcher singleton lock (gateway/status.py).

Verifies that only one process can hold the dispatcher lock at a time,
preventing duplicate dispatchers when two gateway instances race.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_hermes_home(tmp_path, monkeypatch):
    """Redirect get_hermes_home() to a temp dir so locks don't collide."""
    monkeypatch.setattr(
        "gateway.status.get_hermes_home", lambda: tmp_path
    )
    return tmp_path


# ---------------------------------------------------------------------------
# acquire_dispatcher_lock / release_dispatcher_lock
# ---------------------------------------------------------------------------

class TestDispatcherLock:
    """Unit tests for the dispatcher lock primitives."""

    def test_acquire_and_release(self, isolated_hermes_home):
        from gateway.status import (
            acquire_dispatcher_lock,
            release_dispatcher_lock,
            _dispatcher_lock_handle,
        )
        # Ensure clean state
        import gateway.status as gs
        gs._dispatcher_lock_handle = None

        assert acquire_dispatcher_lock() is True
        lock_path = isolated_hermes_home / "dispatcher.lock"
        assert lock_path.exists()

        # Verify PID metadata was written
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()

        release_dispatcher_lock()
        # After release, the handle should be cleared
        assert gs._dispatcher_lock_handle is None

    def test_second_acquire_fails(self, isolated_hermes_home):
        """A second concurrent acquire should return False."""
        import gateway.status as gs
        gs._dispatcher_lock_handle = None

        assert gs.acquire_dispatcher_lock() is True
        # Second call from the same process sees the handle and returns True
        # (same-process re-entry). Simulate cross-process by clearing the
        # handle but keeping the lock held — this is the real race scenario.
        saved = gs._dispatcher_lock_handle
        gs._dispatcher_lock_handle = None

        # The OS-level lock is still held by `saved`; a new open+flock
        # should fail.
        assert gs.acquire_dispatcher_lock() is False

        # Restore for cleanup
        gs._dispatcher_lock_handle = saved
        gs.release_dispatcher_lock()

    def test_idempotent_release(self, isolated_hermes_home):
        """Releasing when not held is a no-op (no crash)."""
        import gateway.status as gs
        gs._dispatcher_lock_handle = None
        gs.release_dispatcher_lock()  # should not raise

    def test_lock_auto_releases_on_handle_close(self, isolated_hermes_home):
        """If the handle is closed externally, re-acquire succeeds."""
        import gateway.status as gs
        gs._dispatcher_lock_handle = None

        assert gs.acquire_dispatcher_lock() is True
        # Simulate external close (e.g. process crash cleanup)
        gs._dispatcher_lock_handle.close()
        gs._dispatcher_lock_handle = None

        # Should be able to re-acquire
        assert gs.acquire_dispatcher_lock() is True
        gs.release_dispatcher_lock()


class TestDispatcherLockCrossProcess:
    """Verify the lock works across separate file handles (simulating
    two processes)."""

    def test_two_file_handles_compete(self, isolated_hermes_home):
        """Open the lock file twice; first flock wins, second fails."""
        from gateway.status import _try_acquire_file_lock, _release_file_lock

        lock_path = isolated_hermes_home / "dispatcher.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        h1 = open(lock_path, "a+")
        h2 = open(lock_path, "a+")

        assert _try_acquire_file_lock(h1) is True
        assert _try_acquire_file_lock(h2) is False

        _release_file_lock(h1)
        # After release, h2 can acquire
        assert _try_acquire_file_lock(h2) is True
        _release_file_lock(h2)

        h1.close()
        h2.close()
