"""Tests for session state recovery on gateway crash.

When the gateway crashes while agent runs are in-flight, a checkpoint
file records which sessions were active.  On restart, the gateway reads
the checkpoint to identify interrupted sessions with precision, rather
than relying solely on the blunt suspend_recently_active() time window.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from gateway.session import SessionCrashCheckpoint


# ── Helpers ────────────────────────────────────────────────────────────────


def _tmp_checkpoint_path(tmp_path):
    """Return a checkpoint file path inside a temp directory."""
    return str(tmp_path / "agent_checkpoints.json")


# ── SessionCrashCheckpoint tests ──────────────────────────────────────────


class TestSessionCrashCheckpointWriteRead:
    """Checkpoint file is written on agent start and read on restart."""

    def test_write_creates_file_with_session_entry(self, tmp_path):
        cp = SessionCrashCheckpoint(path=_tmp_checkpoint_path(tmp_path))
        cp.mark_running("telegram:123:456", session_id="sess_abc")
        data = json.loads(Path(cp.path).read_text())
        assert "telegram:123:456" in data
        assert data["telegram:123:456"]["session_id"] == "sess_abc"

    def test_mark_completed_removes_entry(self, tmp_path):
        cp = SessionCrashCheckpoint(path=_tmp_checkpoint_path(tmp_path))
        cp.mark_running("telegram:123:456", session_id="sess_abc")
        cp.mark_completed("telegram:123:456")
        data = json.loads(Path(cp.path).read_text())
        assert "telegram:123:456" not in data

    def test_read_returns_active_sessions(self, tmp_path):
        cp = SessionCrashCheckpoint(path=_tmp_checkpoint_path(tmp_path))
        cp.mark_running("telegram:123:456", session_id="sess_abc")
        cp.mark_running("discord:789", session_id="sess_def")
        cp.mark_completed("discord:789")
        active = cp.get_active_sessions()
        assert "telegram:123:456" in active
        assert "discord:789" not in active

    def test_checkpoint_includes_timestamp(self, tmp_path):
        cp = SessionCrashCheckpoint(path=_tmp_checkpoint_path(tmp_path))
        before = time.time()
        cp.mark_running("telegram:123:456", session_id="sess_abc")
        data = json.loads(Path(cp.path).read_text())
        assert data["telegram:123:456"]["started_at"] >= before


class TestSessionCrashCheckpointEdgeCases:
    """Edge cases for crash checkpoint persistence."""

    def test_mark_completed_nonexistent_is_noop(self, tmp_path):
        cp = SessionCrashCheckpoint(path=_tmp_checkpoint_path(tmp_path))
        cp.mark_completed("nonexistent:key")  # Should not raise
        # No file created since there was nothing to remove
        assert not os.path.exists(cp.path) or cp.get_active_sessions() == {}

    def test_file_does_not_exist_on_init(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        cp = SessionCrashCheckpoint(path=path)
        assert cp.get_active_sessions() == {}

    def test_corrupted_file_returns_empty(self, tmp_path):
        path = _tmp_checkpoint_path(tmp_path)
        Path(path).write_text("not valid json{{{")
        cp = SessionCrashCheckpoint(path=path)
        assert cp.get_active_sessions() == {}

    def test_multiple_running_sessions(self, tmp_path):
        cp = SessionCrashCheckpoint(path=_tmp_checkpoint_path(tmp_path))
        for i in range(5):
            cp.mark_running(f"platform:{i}", session_id=f"sess_{i}")
        active = cp.get_active_sessions()
        assert len(active) == 5

    def test_clear_removes_all_entries(self, tmp_path):
        cp = SessionCrashCheckpoint(path=_tmp_checkpoint_path(tmp_path))
        cp.mark_running("a:1", session_id="s1")
        cp.mark_running("b:2", session_id="s2")
        cp.clear()
        assert cp.get_active_sessions() == {}


class TestCrashRecoveryIntegration:
    """Integration: restart detects interrupted sessions from checkpoint."""

    def test_interrupted_sessions_detected_after_crash(self, tmp_path):
        """Simulate: gateway crashes with active agents, then restarts."""
        path = _tmp_checkpoint_path(tmp_path)

        # Phase 1: Gateway running, agents active
        cp1 = SessionCrashCheckpoint(path=path)
        cp1.mark_running("telegram:100:200", session_id="sess_active1")
        cp1.mark_running("discord:300", session_id="sess_active2")
        cp1.mark_completed("telegram:100:200")  # This one finished
        # Simulate crash — checkpoint file remains with discord:300

        # Phase 2: Gateway restarts, reads checkpoint
        cp2 = SessionCrashCheckpoint(path=path)
        interrupted = cp2.get_active_sessions()
        assert "discord:300" in interrupted
        assert "telegram:100:200" not in interrupted
        assert interrupted["discord:300"]["session_id"] == "sess_active2"

    def test_no_interrupted_sessions_on_clean_shutdown(self, tmp_path):
        """After a clean shutdown, checkpoint should be empty."""
        path = _tmp_checkpoint_path(tmp_path)

        cp = SessionCrashCheckpoint(path=path)
        cp.mark_running("telegram:100:200", session_id="sess_1")
        cp.mark_completed("telegram:100:200")  # Clean completion
        # On clean shutdown, clear is called
        cp.clear()

        # Restart reads empty checkpoint
        cp2 = SessionCrashCheckpoint(path=path)
        assert cp2.get_active_sessions() == {}

    def test_stale_stop_entry_cleared_by_clean_shutdown(self, tmp_path):
        """Entries left by /stop (generation-invalidated paths) are purged by
        the clean-shutdown clear() so they do not cause false-positive
        suspensions after a subsequent crash."""
        path = _tmp_checkpoint_path(tmp_path)

        cp = SessionCrashCheckpoint(path=path)
        cp.mark_running("telegram:100:200", session_id="sess_1")
        # Simulate /stop: mark_completed is called directly (not via finally block)
        cp.mark_completed("telegram:100:200")

        # A second session that was stopped but whose entry was cleaned up
        # by _interrupt_and_clear_session
        cp.mark_running("discord:300", session_id="sess_2")
        cp.mark_completed("discord:300")

        # Clean shutdown calls clear() — any residual entries are removed
        cp.clear()

        # After crash on next run, no false-positive suspensions
        cp2 = SessionCrashCheckpoint(path=path)
        assert cp2.get_active_sessions() == {}

    def test_stale_eviction_entry_does_not_persist(self, tmp_path):
        """Entries for stale-evicted agents must be removed so they do not
        cause false-positive suspensions on the next crash restart."""
        path = _tmp_checkpoint_path(tmp_path)

        cp = SessionCrashCheckpoint(path=path)
        cp.mark_running("telegram:100:200", session_id="sess_1")
        # Stale eviction calls mark_completed directly (generation invalidated)
        cp.mark_completed("telegram:100:200")

        active = cp.get_active_sessions()
        assert "telegram:100:200" not in active
