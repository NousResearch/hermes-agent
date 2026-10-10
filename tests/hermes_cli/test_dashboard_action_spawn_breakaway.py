"""Dashboard actions must still spawn when the parent Job Object forbids breakaway (Windows).

``_spawn_hermes_action`` detaches its child with ``windows_detach_flags()``, which includes
``CREATE_BREAKAWAY_FROM_JOB``. When the dashboard itself runs inside a Job Object that refuses
breakaway (the Desktop app's Electron wrapper), ``CreateProcess`` fails with ERROR_ACCESS_DENIED
and the Update / Restart Gateway buttons reported ``[WinError 5] Access is denied`` instead of
starting the action. The spawn must retry without the breakaway flag, the fallback the gateway
respawn paths already use.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli._subprocess_compat import windows_detach_flags, windows_detach_flags_without_breakaway


def _spawn_on_windows(tmp_path, *, breakaway_refused: bool):
    from hermes_cli import web_server_gateway

    child = MagicMock(spec=subprocess.Popen, pid=4242)
    calls: list[dict] = []

    def popen(cmd, **kwargs):
        calls.append(kwargs)
        if breakaway_refused and kwargs.get("creationflags") == windows_detach_flags():
            # Shape of the real failure: PermissionError(errno=13, winerror=5), not a bare errno.
            raise PermissionError(13, "Access is denied", None, 5)
        return child

    web_server_gateway._ACTION_LOG_FILES.setdefault("probe", "probe.log")
    with patch.object(web_server_gateway, "_ACTION_LOG_DIR", tmp_path / "logs"), patch.object(
        web_server_gateway.subprocess, "Popen", side_effect=popen
    ), patch.object(web_server_gateway.sys, "platform", "win32"), patch(
        "hermes_cli._subprocess_compat.IS_WINDOWS", True
    ), patch("hermes_cli.web_server.PROJECT_ROOT", tmp_path):
        proc = web_server_gateway._spawn_hermes_action(["update"], "probe")
    return proc, calls


def test_windows_spawn_uses_breakaway_first(tmp_path):
    proc, calls = _spawn_on_windows(tmp_path, breakaway_refused=False)
    assert proc.pid == 4242
    assert [c["creationflags"] for c in calls] == [windows_detach_flags()]


def test_windows_spawn_retries_without_breakaway_when_job_refuses(tmp_path):
    proc, calls = _spawn_on_windows(tmp_path, breakaway_refused=True)
    assert proc.pid == 4242
    assert [c["creationflags"] for c in calls] == [
        windows_detach_flags(),
        windows_detach_flags_without_breakaway(),
    ]
    # Everything except the detach flag is identical on the retry.
    first, second = calls
    for key in ("cwd", "stdin", "stdout", "stderr", "env"):
        assert first[key] == second[key]


def test_windows_spawn_propagates_unrelated_oserror(tmp_path):
    from hermes_cli import web_server_gateway

    web_server_gateway._ACTION_LOG_FILES.setdefault("probe", "probe.log")
    calls: list[dict] = []

    def popen(cmd, **kwargs):
        calls.append(kwargs)
        raise FileNotFoundError(2, "no such file")

    with patch.object(web_server_gateway, "_ACTION_LOG_DIR", tmp_path / "logs"), patch.object(
        web_server_gateway.subprocess, "Popen", side_effect=popen
    ), patch.object(web_server_gateway.sys, "platform", "win32"), patch(
        "hermes_cli._subprocess_compat.IS_WINDOWS", True
    ), patch("hermes_cli.web_server.PROJECT_ROOT", tmp_path), pytest.raises(FileNotFoundError):
        web_server_gateway._spawn_hermes_action(["update"], "probe")
    # No doomed second attempt that would mask the first error.
    assert len(calls) == 1


def test_windows_spawn_does_not_retry_non_breakaway_permission_error(tmp_path):
    """A PermissionError that is NOT ERROR_ACCESS_DENIED from the job (e.g. ACL on the exe) is re-raised."""
    from hermes_cli import web_server_gateway

    web_server_gateway._ACTION_LOG_FILES.setdefault("probe", "probe.log")
    calls: list[dict] = []

    def popen(cmd, **kwargs):
        calls.append(kwargs)
        raise PermissionError(13, "Access is denied")  # winerror is None here

    with patch.object(web_server_gateway, "_ACTION_LOG_DIR", tmp_path / "logs"), patch.object(
        web_server_gateway.subprocess, "Popen", side_effect=popen
    ), patch.object(web_server_gateway.sys, "platform", "win32"), patch(
        "hermes_cli._subprocess_compat.IS_WINDOWS", True
    ), patch("hermes_cli.web_server.PROJECT_ROOT", tmp_path), pytest.raises(PermissionError):
        web_server_gateway._spawn_hermes_action(["update"], "probe")
    assert len(calls) == 1
