"""Tests for terminate_pid Windows escalation logic.

Verifies that terminate_pid's force=False path escalates from os.kill to
taskkill /T /F when Windows reports ERROR_ACCESS_DENIED on the target PID
(observed when the target is an orphaned job-object child left behind after
the Hermes gateway crashes on Windows — the parent cmd.exe wrapper dies and
the child python.exe becomes un-reapable by TerminateProcess).

Without the escalation, the gateway restart manager's --replace path
bails out at the PermissionError catch and returns False, leaving every
Hermes gateway crash on Windows unrecoverable until an operator manually
kills the orphan.

These tests are Windows-only (the escalation is a no-op on POSIX and
cannot be exercised under monkeypatching because the production
windows_hide_flags() returns CREATE_NO_WINDOW = 0x08000000 only on real
Windows; patching it would diverge from the actual wire format).
"""

from unittest.mock import MagicMock

import pytest

from gateway import status

pytestmark = pytest.mark.windows_only


@pytest.fixture
def windows_kill_env(monkeypatch):
    """Set up a Windows environment for terminate_pid escalation tests.

    Tracks os.kill calls and stub-runs taskkill so we can assert the
    escalation behavior. Patches status.os.kill and status.subprocess.run;
    the latter is what the escalation falls back to after PermissionError.
    The kill behavior (raise / return) and the taskkill return code can be
    overridden by passing kwargs to the fixture.
    """

    state = {"os_kill_calls": [], "taskkill_calls": [], "raise_on_kill": None, "taskkill_returncode": 0}

    def _mock_os_kill(pid, sig):
        state["os_kill_calls"].append((pid, sig))
        if state["raise_on_kill"] is not None:
            raise state["raise_on_kill"]

    def _mock_subprocess_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        state["taskkill_calls"].append(list(cmd))
        result = MagicMock()
        result.returncode = state["taskkill_returncode"]
        result.stderr = ""
        result.stdout = ""
        return result

    monkeypatch.setattr(status.os, "kill", _mock_os_kill)
    monkeypatch.setattr(status, "subprocess", MagicMock(run=_mock_subprocess_run))
    return state


def _raise_kill(state, exc):
    state["raise_on_kill"] = exc
    state["os_kill_calls"] = []
    state["taskkill_calls"] = []


def test_force_false_permission_error_escalates_to_taskkill(windows_kill_env):
    """PermissionError on os.kill -> taskkill /T /F, no exception propagated."""
    _raise_kill(windows_kill_env, PermissionError("access denied"))

    status.terminate_pid(1234, force=False)

    assert len(windows_kill_env["os_kill_calls"]) == 1
    assert len(windows_kill_env["taskkill_calls"]) == 1
    assert windows_kill_env["taskkill_calls"][0] == ["taskkill", "/PID", "1234", "/T", "/F"]


def test_force_false_os_error_does_not_escalate(windows_kill_env):
    """Non-PermissionError OSError is still propagated to caller (no escalation)."""
    _raise_kill(windows_kill_env, OSError("process not found"))

    with pytest.raises(OSError):
        status.terminate_pid(1234, force=False)

    assert len(windows_kill_env["os_kill_calls"]) == 1
    assert len(windows_kill_env["taskkill_calls"]) == 0


def test_force_false_success_does_not_call_taskkill(windows_kill_env):
    """When os.kill succeeds on Windows, no taskkill escalation happens."""
    _raise_kill(windows_kill_env, None)

    status.terminate_pid(1234, force=False)

    assert len(windows_kill_env["os_kill_calls"]) == 1
    assert len(windows_kill_env["taskkill_calls"]) == 0


def test_force_false_taskkill_fallback_failure_reraises(windows_kill_env):
    """When taskkill also fails after os.kill PermissionError, OSError is raised."""
    _raise_kill(windows_kill_env, PermissionError("access denied"))
    windows_kill_env["taskkill_returncode"] = 128
    windows_kill_env["taskkill_calls"] = []

    with pytest.raises(OSError):
        status.terminate_pid(1234, force=False)

    assert len(windows_kill_env["os_kill_calls"]) == 1
    assert len(windows_kill_env["taskkill_calls"]) == 1
