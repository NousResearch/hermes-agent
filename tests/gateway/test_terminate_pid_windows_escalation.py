"""Tests for terminate_pid Windows escalation logic.

Verifies the fix that escalates os.kill(SIGTERM) -> taskkill /T /F when
the target is an orphaned job-object child on Windows and TerminateProcess
returns ERROR_ACCESS_DENIED (PermissionError).

Background: when the Hermes gateway crashes on Windows, the parent
cmd.exe wrapper dies and the child python.exe becomes an orphaned
job-object process. Windows refuses TerminateProcess on it (Permission
denied), and the restart manager's --replace path bails out at line 30043
of gateway/run.py with return False. Without this escalation, every
gateway crash on Windows is unrecoverable until an operator manually
kills the orphan.
"""

import sys
from unittest.mock import MagicMock

import pytest

from gateway import status


def _patch_windows(monkeypatch, *, os_kill_raises=None, taskkill_returncode=0):
    """Set up a Windows environment for terminate_pid."""
    # Force _IS_WINDOWS = True
    monkeypatch.setattr(status, "_IS_WINDOWS", True)
    monkeypatch.setattr(status, "signal", sys.modules["signal"])

    # Track os.kill and taskkill calls
    os_kill_calls = []
    taskkill_calls = []

    def _mock_os_kill(pid, sig):
        os_kill_calls.append((pid, sig))
        if os_kill_raises is not None:
            raise os_kill_raises

    def _mock_subprocess_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        taskkill_calls.append(list(cmd))
        result = MagicMock()
        result.returncode = taskkill_returncode
        result.stderr = ""
        result.stdout = ""
        return result

    monkeypatch.setattr(status.os, "kill", _mock_os_kill)
    monkeypatch.setattr(status, "subprocess", MagicMock(run=_mock_subprocess_run))

    return os_kill_calls, taskkill_calls


def test_force_false_windows_permission_error_escalates_to_taskkill(monkeypatch):
    """When os.kill raises PermissionError on Windows, terminate_pid escalates to taskkill /T /F."""
    os_kill_calls, taskkill_calls = _patch_windows(
        monkeypatch, os_kill_raises=PermissionError("access denied")
    )

    # Should NOT raise — the escalation handles it
    status.terminate_pid(1234, force=False)

    assert len(os_kill_calls) == 1
    assert len(taskkill_calls) == 1
    # Verify the escalation was the correct taskkill command
    assert taskkill_calls[0] == ["taskkill", "/PID", "1234", "/T", "/F"]


def test_force_false_windows_os_error_does_not_escalate(monkeypatch):
    """On Windows, OSError (non-PermissionError) is still propagated to caller."""
    os_kill_calls, taskkill_calls = _patch_windows(
        monkeypatch, os_kill_raises=OSError("process not found")
    )

    with pytest.raises(OSError):
        status.terminate_pid(1234, force=False)

    assert len(os_kill_calls) == 1
    assert len(taskkill_calls) == 0  # No escalation


def test_force_false_windows_success_does_not_call_taskkill(monkeypatch):
    """When os.kill succeeds on Windows, no taskkill escalation happens."""
    os_kill_calls, taskkill_calls = _patch_windows(monkeypatch, os_kill_raises=None)

    status.terminate_pid(1234, force=False)

    assert len(os_kill_calls) == 1
    assert len(taskkill_calls) == 0


def test_force_false_windows_taskkill_fallback_failure_reraises(monkeypatch):
    """When taskkill also fails after os.kill PermissionError, the OSError is raised."""
    os_kill_calls, taskkill_calls = _patch_windows(
        monkeypatch,
        os_kill_raises=PermissionError("access denied"),
        taskkill_returncode=128,
    )

    with pytest.raises(OSError):
        status.terminate_pid(1234, force=False)

    assert len(os_kill_calls) == 1
    assert len(taskkill_calls) == 1
