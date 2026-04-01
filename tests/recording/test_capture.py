"""Tests for recording/capture.py — session capture singleton."""

import pytest
from unittest.mock import patch

import recording.store as store_mod
from recording.capture import RecordingSession, get_active_session, _lock
import recording.capture as capture_mod


@pytest.fixture(autouse=True)
def _reset_global_session():
    """Reset the global singleton between tests."""
    capture_mod._active_session = None
    yield
    capture_mod._active_session = None


@pytest.fixture(autouse=True)
def _patch_recordings_dir(tmp_path, monkeypatch):
    """Ensure RECORDINGS_DIR points to the test-local temp directory."""
    rec_dir = tmp_path / "hermes_test" / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store_mod, "RECORDINGS_DIR", rec_dir)


class TestGetActiveSession:
    def test_none_when_no_recording(self):
        assert get_active_session() is None


class TestRecordingSession:
    def test_start_stop_lifecycle(self):
        session = RecordingSession("lifecycle-test", "A test")
        assert not session.is_active

        session.start()
        assert session.is_active
        assert get_active_session() is session

        summary = session.stop()
        assert not session.is_active
        assert get_active_session() is None
        assert summary["name"] == "lifecycle-test"
        assert summary["step_count"] == 0

    def test_capture_calls(self):
        session = RecordingSession("capture-test")
        session.start()

        session.capture_tool_call("terminal", {"command": "echo hi"}, "hi", True)
        session.capture_tool_call("write_file", {"path": "/tmp/x"}, "ok", True)

        summary = session.stop()
        assert summary["step_count"] == 2

        from recording.store import get_recording
        rec = get_recording("capture-test")
        assert len(rec["steps"]) == 2
        assert rec["steps"][0]["tool"] == "terminal"
        assert rec["steps"][1]["tool"] == "write_file"

    def test_capture_when_inactive(self):
        session = RecordingSession("inactive-test")
        # Don't start the session
        session.capture_tool_call("terminal", {"command": "echo"}, "out", True)
        # Should be a no-op — step count stays 0
        assert session._step_count == 0

    def test_double_start_raises(self):
        s1 = RecordingSession("first")
        s1.start()

        s2 = RecordingSession("second")
        with pytest.raises(RuntimeError, match="already active"):
            s2.start()

        s1.stop()

    def test_stop_without_start(self):
        session = RecordingSession("never-started")
        # stop() should work gracefully even if never started
        summary = session.stop()
        assert summary["step_count"] == 0

    def test_start_existing_recording(self):
        """Starting a session for an already-existing recording should resume it."""
        from recording.store import create_recording, add_step
        create_recording("existing")
        add_step("existing", "terminal", {"command": "old"}, "old-out", True)

        session = RecordingSession("existing")
        session.start()
        session.capture_tool_call("terminal", {"command": "new"}, "new-out", True)
        session.stop()

        from recording.store import get_recording
        rec = get_recording("existing")
        assert len(rec["steps"]) == 2
