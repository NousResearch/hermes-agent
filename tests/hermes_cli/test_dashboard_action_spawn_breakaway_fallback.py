"""``_spawn_hermes_action`` must survive a parent Job Object that forbids breakaway.

Windows rejects ``CREATE_BREAKAWAY_FROM_JOB`` with ``ERROR_ACCESS_DENIED`` (``winerror`` 5) when
the dashboard's own job object was created without ``JOB_OBJECT_LIMIT_BREAKAWAY_OK`` — a Windows
service, Task Scheduler, or some RDP/console hosts. ``windows_detach_flags()`` documents the
contract ("A job that forbids breakaway yields PermissionError from Popen — callers catch OSError
and fall back to ``windows_detach_flags_without_breakaway``"), and ``gateway.py`` /
``main_desktop.py`` honour it; the dashboard action spawn did not. So ``POST /api/hermes/update``
answered ``500 Failed to start update: [WinError 5] Access is denied.`` instead of starting the
update, and every other dashboard action sharing this spawn (gateway restart, doctor, backup, …)
failed the same way.

Behavioral: drives the real ``_spawn_hermes_action`` with a mocked ``subprocess.Popen``; the two
collaborators it reaches for (``web_server.PROJECT_ROOT`` and ``_launchers.runtime_command``) are
stubbed so the unit stays importable without the dashboard's web stack.
``IS_WINDOWS`` is patched next to ``sys.platform`` so the creation flags are real on every lane —
a Windows-only skip would leave the assertions below unexecuted on the others.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

BREAKAWAY_BIT = 0x01000000  # CREATE_BREAKAWAY_FROM_JOB


def _windows_access_denied() -> OSError:
    """The exception ``CreateProcess`` raises for a denied job breakaway.

    Built explicitly: ``winerror`` is a Windows-only attribute of ``OSError``, so a bare
    ``OSError(5, …)`` would look like an unrelated failure on the POSIX lanes.
    """
    exc = OSError(5, "Access is denied")  # ERROR_ACCESS_DENIED
    exc.winerror = 5
    return exc


def _prepare(monkeypatch, tmp_path):
    """Import the real spawn module and neutralise everything but the Popen call itself."""
    import hermes_cli.web_server_gateway as wsg

    server = types.ModuleType("hermes_cli.web_server")
    server.PROJECT_ROOT = tmp_path
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", server)

    launchers = types.ModuleType("hermes_cli._launchers")
    launchers.runtime_command = lambda root, subcommand: ["hermes", *subcommand]
    monkeypatch.setitem(sys.modules, "hermes_cli._launchers", launchers)

    monkeypatch.setattr(wsg, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(wsg, "_ACTION_LOG_DIR", tmp_path)
    monkeypatch.setattr(
        wsg, "_ACTION_LOG_FILES", {"hermes-update": "hermes-update.log", "doctor": "doctor.log"})
    monkeypatch.setattr(wsg, "_profile_action_environment", lambda subcommand, env=None: {"PATH": "/usr/bin"})
    monkeypatch.setattr(wsg, "_action_targets_system_gateway", lambda subcommand: False)

    # Real flags on any host: ``_subprocess_compat`` reads this global at call time.
    import hermes_cli._subprocess_compat as sc

    monkeypatch.setattr(sc, "IS_WINDOWS", True)
    return wsg


def test_action_spawn_retries_without_breakaway_when_the_job_forbids_it(monkeypatch, tmp_path):
    from hermes_cli._subprocess_compat import windows_detach_flags, windows_detach_flags_without_breakaway

    wsg = _prepare(monkeypatch, tmp_path)
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if len(calls) == 1:
            raise _windows_access_denied()
        return MagicMock(pid=4242)

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    proc = wsg._spawn_hermes_action(["update"], "hermes-update")

    assert len(calls) == 2, "a denied breakaway must be retried exactly once, never surfaced"
    (cmd1, kw1), (cmd2, kw2) = calls

    # The same action argv and the same spawn configuration on both attempts: only the flags move.
    assert cmd1 == cmd2 == ["hermes", "update"]
    assert kw1["cwd"] == kw2["cwd"]
    assert kw1["env"] is kw2["env"]
    assert kw1["stdin"] is kw2["stdin"]
    assert kw1["stderr"] is kw2["stderr"]
    assert kw1["stdout"] is kw2["stdout"], "the retry must reuse the action log fd, not reopen it"
    assert "start_new_session" not in kw2

    # The point of the fallback: the primary asks for breakaway, the retry drops exactly that bit.
    assert kw1["creationflags"] == windows_detach_flags()
    assert kw1["creationflags"] & BREAKAWAY_BIT, "primary spawn must request CREATE_BREAKAWAY_FROM_JOB"
    assert kw2["creationflags"] == windows_detach_flags_without_breakaway()
    assert not kw2["creationflags"] & BREAKAWAY_BIT, "retry must drop CREATE_BREAKAWAY_FROM_JOB"

    # The successful retry is what gets recorded as the action's process.
    assert proc.pid == 4242
    assert wsg._ACTION_PROCS["hermes-update"] is proc


def test_action_spawn_reraises_unrelated_oserror(monkeypatch, tmp_path):
    """Only a denied job breakaway is retried; any other spawn failure stays a single clear 500."""
    wsg = _prepare(monkeypatch, tmp_path)
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        raise OSError(2, "The system cannot find the file specified")  # ERROR_FILE_NOT_FOUND

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    with pytest.raises(OSError) as excinfo:
        wsg._spawn_hermes_action(["update"], "hermes-update")

    assert excinfo.value.errno == 2
    assert len(calls) == 1, "a non-breakaway failure must not be masked by a doomed second attempt"


def test_action_spawn_keeps_posix_behaviour(monkeypatch, tmp_path):
    """POSIX keeps ``start_new_session``; the Windows branch must not leak onto it."""
    wsg = _prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(wsg, "sys", SimpleNamespace(platform="linux"))
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return MagicMock(pid=4242)

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    wsg._spawn_hermes_action(["update"], "hermes-update")

    assert len(calls) == 1
    _cmd, kwargs = calls[0]
    assert kwargs["start_new_session"] is True
    assert "creationflags" not in kwargs
