"""LIVE Windows: the bootstrap's relaunched interpreter must be able to start a detached gateway.

hermes.exe runs the bootstrap interpreter in a KILL_ON_JOB_CLOSE | SILENT_BREAKAWAY_OK job. A plain
``subprocess.call`` child silently breaks away into the nearest enclosing job that forbids silent
breakaway; if that job also forbids breakaway (the Multica daemon's, an agent host's), the relaunched
process's CREATE_BREAKAWAY_FROM_JOB spawn fails with WinError 5 and the gateway start refuses with
``windows_breakaway_unavailable``. Every Multica per-task home is relaunched, so the Hermes runtime
never started a gateway there.

Topology (real job objects, real processes): enclosing job (KILL_ON_JOB_CLOSE only: no breakaway)
> launcher job (0x3000) > driver = the bootstrap > relaunched probe, which spawns with
``windows_detach_flags()`` exactly as ``spawn_unmanaged_gateway`` does.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")  # native Windows only

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CREATE_SUSPENDED = 0x00000004
_PROBE_DENIED = 3
# The base interpreter, never a venv's Scripts\python.exe: that redirector puts the real interpreter
# in a launcher-like job of its own and would hide the failure (PM's store python has no redirector).
_PYTHON = getattr(sys, "_base_executable", None) or sys.executable

_PROBE = (
    f"import subprocess, sys; sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
    "from hermes_cli._subprocess_compat import windows_detach_flags\n"
    "try:\n"
    "    subprocess.Popen([sys.executable, '-c', 'pass'], creationflags=windows_detach_flags(),\n"
    "                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait(30)\n"
    "except PermissionError:\n"
    f"    raise SystemExit({_PROBE_DENIED})\n"
)


class _Basic(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _Extended(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _Basic), ("IoInfo", ctypes.c_ulonglong * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k.SetInformationJobObject.restype = wintypes.BOOL
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k.AssignProcessToJobObject.restype = wintypes.BOOL
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CloseHandle.restype = wintypes.BOOL
    return k


def _job(k, flags: int) -> int:
    job = k.CreateJobObjectW(None, None)
    assert job, f"CreateJobObjectW failed (winerror={ctypes.get_last_error()})"
    info = _Extended()
    info.BasicLimitInformation.LimitFlags = flags
    assert k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)), ctypes.get_last_error()
    return job


def _bootstrap_exit_code(relaunch: str) -> int:
    """Run a bootstrap stand-in under hermes.exe's job, inside a no-breakaway job; return its status.

    ``relaunch`` names the call the stand-in relaunches its probe with (``module.function``)."""
    from hermes_cli.update_lock import resume_suspended_child

    k = _kernel32()
    enclosing = _job(k, 0x2000)  # KILL_ON_JOB_CLOSE alone: forbids breakaway and silent breakaway
    launcher = _job(k, 0x2000 | 0x1000)  # what hermes.exe gives the bootstrap interpreter
    module, _, function = relaunch.rpartition(".")
    driver_source = (
        f"import importlib, sys; sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        f"relaunch = getattr(importlib.import_module({module!r}), {function!r})\n"
        f"raise SystemExit(relaunch([sys.executable, '-I', '-c', {_PROBE!r}]))\n"
    )
    try:
        driver = subprocess.Popen([_PYTHON, "-I", "-c", driver_source], cwd=str(_REPO_ROOT),
                                  creationflags=_CREATE_SUSPENDED)
        for job in (enclosing, launcher):  # enclosing first: the launcher job nests inside it
            if not k.AssignProcessToJobObject(job, int(driver._handle)):
                error = ctypes.get_last_error()
                driver.kill()
                pytest.fail(f"AssignProcessToJobObject failed (winerror={error})")
        resume_suspended_child(driver)
        return driver.wait(timeout=120)
    finally:
        k.CloseHandle(launcher)
        k.CloseHandle(enclosing)  # kill-on-close reaps anything the run left behind


def test_plain_relaunch_cannot_start_detached_gateway():
    """The mechanism the fix answers: a bare subprocess.call child lands in the enclosing job."""
    assert _bootstrap_exit_code("subprocess.call") == _PROBE_DENIED


def test_relaunch_keeps_launcher_job_so_detached_gateway_starts():
    assert _bootstrap_exit_code("hermes_cli.venv_sync.call_relaunched") == 0
