"""The WhatsApp bridge still starts when the gateway runs inside a job that forbids breakaway.

A gateway started by the Desktop app, a scheduled task or a service wrapper runs inside a Windows job
object without JOB_OBJECT_LIMIT_BREAKAWAY_OK. ``CREATE_BREAKAWAY_FROM_JOB`` then makes the spawn fail with
WinError 5, so the bridge must be retried once without that bit instead of failing on every reconnect.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from plugins.platforms.whatsapp import adapter

_BREAKAWAY = 0x01000000
_WITH = 0x00000200 | 0x08000000 | _BREAKAWAY
_WITHOUT = 0x00000200 | 0x08000000


def _access_denied() -> OSError:
    exc = PermissionError(13, "Access is denied")
    exc.winerror = 5
    return exc


def _patch(monkeypatch, *errors):
    """Patch the spawn so that its n-th call raises errors[n] (None means it succeeds)."""
    pending, calls = list(errors), []

    def fake_spawn_owned(argv, *, purpose, **kwargs):
        calls.append(kwargs["creationflags"])
        error = pending.pop(0) if pending else None
        if error is not None:
            raise error
        return f"proc-{len(calls)}"

    monkeypatch.setattr(adapter, "spawn_owned", fake_spawn_owned)
    monkeypatch.setattr(adapter, "windows_detach_flags", lambda: _WITH)
    monkeypatch.setattr(adapter, "windows_detach_flags_without_breakaway", lambda: _WITHOUT)
    return calls


def test_retries_without_breakaway_when_the_job_forbids_it(monkeypatch, caplog):
    calls = _patch(monkeypatch, _access_denied())

    with caplog.at_level("WARNING", logger=adapter.logger.name):
        proc = adapter._spawn_bridge_process(["node", "bridge.js"], log_fh=None, env={})

    assert proc == "proc-2"
    assert calls == [_WITH, _WITHOUT]
    assert "retrying without it" in caplog.text


def test_spawns_once_when_breakaway_is_allowed(monkeypatch):
    calls = _patch(monkeypatch)

    assert adapter._spawn_bridge_process(["node", "bridge.js"], log_fh=None, env={}) == "proc-1"
    assert calls == [_WITH]


def test_other_failures_are_not_retried(monkeypatch):
    missing = FileNotFoundError(2, "The system cannot find the file specified")
    missing.winerror = 2
    calls = _patch(monkeypatch, missing)
    with pytest.raises(FileNotFoundError):
        adapter._spawn_bridge_process(["node", "bridge.js"], log_fh=None, env={})
    assert calls == [_WITH]

    # errno 5 on Windows is EIO, not access denied: without winerror 5 there is no retry
    calls = _patch(monkeypatch, OSError(5, "I/O error"))
    with pytest.raises(OSError):
        adapter._spawn_bridge_process(["node", "bridge.js"], log_fh=None, env={})
    assert calls == [_WITH]


def test_a_second_refusal_is_raised(monkeypatch):
    calls = _patch(monkeypatch, _access_denied(), _access_denied())

    with pytest.raises(PermissionError):
        adapter._spawn_bridge_process(["node", "bridge.js"], log_fh=None, env={})
    assert calls == [_WITH, _WITHOUT]


def test_the_retry_keeps_the_bridge_stdio_and_umask(monkeypatch):
    seen = []

    def fake_spawn_owned(argv, *, purpose, **kwargs):
        seen.append((purpose, kwargs))
        if len(seen) == 1:
            raise _access_denied()
        return "proc"

    monkeypatch.setattr(adapter, "spawn_owned", fake_spawn_owned)
    log = object()
    adapter._spawn_bridge_process(["node", "bridge.js"], log_fh=log, env={"A": "1"}, umask=0o077)

    for purpose, kwargs in seen:
        assert purpose == "whatsapp-bridge"
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stdout"] is log and kwargs["stderr"] is log
        assert kwargs["env"] == {"A": "1"} and kwargs["umask"] == 0o077


# The real failure: a process inside a job object that forbids breakaway. Windows only.

_CHILD = r"""
import ctypes, json, os, subprocess, sys
from ctypes import wintypes

sys.path.insert(0, sys.argv[1])
k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class Basic(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class Io(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class Extended(ctypes.Structure):
    _fields_ = [("Basic", Basic), ("Io", Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


k32.CreateJobObjectW.restype = wintypes.HANDLE
k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
k32.GetCurrentProcess.restype = wintypes.HANDLE

# LimitFlags 0: neither BREAKAWAY_OK nor SILENT_BREAKAWAY_OK, like the Desktop app's job
job = k32.CreateJobObjectW(None, None)
limits = Extended()
if (not job
        or not k32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
        or not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess())):
    print(json.dumps({"setup_error": ctypes.get_last_error()}))
    raise SystemExit(0)

from hermes_cli._subprocess_compat import windows_detach_flags
from plugins.platforms.whatsapp import adapter

argv = [sys.executable, "-c", "pass"]


def attempt(spawn):
    try:
        return {"ok": True, "returncode": spawn().wait(30)}
    except OSError as exc:
        return {"ok": False, "winerror": getattr(exc, "winerror", None)}


print(json.dumps({
    "plain": attempt(lambda: subprocess.Popen(argv, creationflags=windows_detach_flags())),
    "bridge": attempt(lambda: adapter._spawn_bridge_process(argv, None, dict(os.environ))),
}))
"""


@pytest.mark.platforms("windows")
def test_bridge_starts_inside_a_job_that_forbids_breakaway(tmp_path):
    root = str(Path(__file__).resolve().parents[2])
    done = subprocess.run([sys.executable, "-c", _CHILD, root], capture_output=True, text=True, timeout=120,
                          check=False, env={**os.environ, "HERMES_HOME": str(tmp_path)})
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    if "setup_error" in out:
        pytest.skip(f"could not create a job object here (error {out['setup_error']})")
    # the control: in this job a plain breakaway spawn really is refused, which is what the gateway hits
    assert out["plain"] == {"ok": False, "winerror": 5}, out
    assert out["bridge"] == {"ok": True, "returncode": 0}, out
