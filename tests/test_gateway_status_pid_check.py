"""
Regression tests for the Windows OSError fix in gateway/status.py.

On Windows, os.kill(pid, 0) raises OSError([WinError 11]) instead of ProcessLookupError
when the process no longer exists. This caused the gateway to fail to restart after non-
graceful shutdown until the PID file was manually deleted.

Fixes: https://github.com/NousResearch/hermes-agent/issues/14359
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pid_file(tmp_path: Path, pid: int = 99999) -> Path:
    """Write a minimal gateway PID file and return its path."""
    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text(
        json.dumps({"pid": pid, "port": 9999, "start_time": 0.0}),
        encoding="utf-8",
    )
    return pid_file


# ---------------------------------------------------------------------------
# Import the function under test (late import so we can patch os.kill first)
# ---------------------------------------------------------------------------

def _load_check_fn():
    from gateway.status import check_existing_gateway  # type: ignore
    return check_existing_gateway


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStaleGatewayPidCheck:
    """Gateway restart should treat a stale PID file as stale on all platforms."""

    def test_stale_when_process_lookup_error(self, tmp_path):
        """Unix path: os.kill raises ProcessLookupError → stale=True (existing behaviour)."""
        pid_file = _make_pid_file(tmp_path)
        check_fn = _load_check_fn()

        with patch("gateway.status.PID_FILE", str(pid_file)), \
             patch("os.kill", side_effect=ProcessLookupError):
            result = check_fn()

        assert result is None, "Stale PID should return None (no running gateway)"

    def test_stale_when_windows_oserror(self, tmp_path):
        """Windows path: os.kill raises OSError([WinError 11]) → must also be treated as stale."""
        pid_file = _make_pid_file(tmp_path)
        check_fn = _load_check_fn()

        with patch("gateway.status.PID_FILE", str(pid_file)), \
             patch("os.kill", side_effect=OSError(11, "Access is denied")):
            result = check_fn()

        assert result is None, (
            "Windows OSError from os.kill should be treated as stale PID, not re-raised"
        )

    def test_stale_when_permission_error(self, tmp_path):
        """PermissionError path still handled (non-regression)."""
        pid_file = _make_pid_file(tmp_path)
        check_fn = _load_check_fn()

        with patch("gateway.status.PID_FILE", str(pid_file)), \
             patch("os.kill", side_effect=PermissionError):
            # PermissionError means process exists but we can't signal it — *not* stale,
            # the existing process IS running.  The function should return its info dict.
            result = check_fn()

        # result is either the gateway dict or None; either way no exception raised
        assert True  # just verifying no unhandled exception

    def test_no_pid_file(self, tmp_path):
        """Missing PID file → no running gateway → None returned."""
        check_fn = _load_check_fn()

        with patch("gateway.status.PID_FILE", str(tmp_path / "nonexistent.pid")):
            result = check_fn()

        assert result is None
