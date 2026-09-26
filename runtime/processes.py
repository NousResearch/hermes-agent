"""Shared process containment primitives.

This module owns host-process lifecycle mechanisms that are independent of CLI orchestration.
The owner-process job and child-containment job intentionally remain separate: they have
different Windows breakaway semantics.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from ctypes import wintypes
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from typing import Optional


logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"

# Module-global job handle: must live exactly as long as this process so the
# kernel closes it (and kills the job) when we die. Never close it manually.
_JOB_HANDLE = None


def attach_self_to_kill_on_close_job() -> bool:
    """Place this process in a job that dies with it. Windows-only and idempotent.

    BREAKAWAY_OK keeps children spawned with CREATE_BREAKAWAY_FROM_JOB (gateway relaunch
    during update, detached watchers) escaping exactly as before.
    """
    global _JOB_HANDLE
    if not _IS_WINDOWS or _JOB_HANDLE is not None:
        return _JOB_HANDLE is not None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x0800
        JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x1000
        JobObjectExtendedLimitInformation = 9

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(wintypes.ULONG)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                *((n, ctypes.c_size_t) for n in (
                    "ProcessMemoryLimit", "JobMemoryLimit",
                    "PeakProcessMemoryUsed", "PeakJobMemoryUsed")),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return False
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_BREAKAWAY_OK
            | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
        )
        ok = kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        )
        if not ok or not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            kernel32.CloseHandle(job)
            return False
        _JOB_HANDLE = job
        logger.debug("attached to kill-on-close job object")
        return True
    except Exception:
        logger.debug("job object self-attach failed", exc_info=True)
        return False


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _WindowsJob:
    """Owner-held, non-breakaway job for one spawned process tree."""

    def __init__(self):
        self._lock = threading.Lock()
        self._api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            (
                "SetInformationJobObject",
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
                wintypes.BOOL,
            ),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ):
            fn = getattr(self._api, name)
            fn.argtypes = args
            fn.restype = result

        # NULL security attributes create a non-inheritable, unnamed owner handle.
        self._handle = self._api.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = _ExtendedLimits()
            # Neither BREAKAWAY_OK nor SILENT_BREAKAWAY_OK: descendants stay contained.
            limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
            if not self._api.SetInformationJobObject(
                self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise

    def assign(self, proc: subprocess.Popen) -> None:
        # Popen retains the original process handle, avoiding a PID-reuse race.
        if not self._api.AssignProcessToJobObject(self._handle, int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        """Terminate the contained tree; repeated closes are harmless."""
        with self._lock:
            if self._handle is not None:
                if not self._api.CloseHandle(self._handle):
                    raise ctypes.WinError(ctypes.get_last_error())
                self._handle = None


def spawn_contained_process(cmd, **kwargs) -> tuple[subprocess.Popen, _WindowsJob | None]:
    """Start a process and return an owner-held containment handle when available.

    Windows creates the child suspended, assigns it to a non-breakaway
    KILL_ON_JOB_CLOSE job, and resumes it only after successful assignment.
    Other hosts retain Popen's ordinary behavior.
    """
    if sys.platform != "win32":
        return subprocess.Popen(cmd, **kwargs), None

    # Keep psutil off the import path for callers that never spawn a Windows child.
    import psutil

    job = _WindowsJob()
    proc = None
    try:
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | 0x00000004  # CREATE_SUSPENDED
        proc = subprocess.Popen(cmd, **kwargs)
        job.assign(proc)
        psutil.Process(proc.pid).resume()
        return proc, job
    except BaseException:
        try:
            if proc is not None:
                # Assignment may have failed: closing an empty job is not enough.
                proc.kill()
                proc.wait(timeout=10)
        finally:
            try:
                job.close()
            finally:
                if proc is not None:
                    for stream in (proc.stdin, proc.stdout, proc.stderr):
                        if stream is not None:
                            stream.close()
                    proc._handle.Close()
        raise


@contextmanager
def _process_tree_snapshot(pid: int, *, hard_kill: bool):
    """Stop each hard-kill target before discovering its children: a running
    parent can fork after psutil builds its PID map and escape the final signal.
    Resume anything we stopped if signalling fails. Graceful signals never stop
    their recipients, since their handlers must remain able to run.
    """
    descendants = []
    stopped = []
    try:
        try:
            import psutil
            root = psutil.Process(pid)
            descendants = root.children(recursive=True)
            if hard_kill:
                pending = [root]
                seen = {os.getpid()}
                known = {process.pid: process for process in descendants}
                stop_deadline = time.monotonic() + 1.0
                while pending:
                    for process in pending:
                        if process.pid in seen:
                            continue
                        seen.add(process.pid)
                        if time.monotonic() >= stop_deadline:
                            raise TimeoutError("process tree did not stop before snapshot deadline")
                        try:
                            status = process.status()
                            if status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                                continue
                            if status != psutil.STATUS_STOPPED:
                                process.suspend()
                                stopped.append(process)
                            while process.status() != psutil.STATUS_STOPPED:
                                if time.monotonic() >= stop_deadline:
                                    raise TimeoutError("process did not stop before snapshot deadline")
                                time.sleep(0.001)
                        except psutil.NoSuchProcess:
                            continue
                    # Rescan after stopping the discovered generation. A child
                    # may have forked while that generation was being stopped.
                    for process in root.children(recursive=True):
                        known.setdefault(process.pid, process)
                    descendants = list(known.values())
                    pending = [process for process in descendants if process.pid not in seen]
        except Exception:
            # Preserve the existing best-effort group fallback when discovery or
            # stopping is unavailable; never strand a successfully stopped target.
            logger.debug("kill_process_tree: snapshot incomplete for pid %s", pid, exc_info=True)
        yield descendants
    finally:
        for process in stopped:
            try:
                process.resume()
            except Exception:
                logger.debug("kill_process_tree: target already gone or resume refused", exc_info=True)


def kill_process_tree(pid: int, *, sig: Optional[int] = None) -> bool:
    """Terminate ``pid`` and all its descendants, portably; True when anything was signalled.

    Windows: ``taskkill /F /T`` (``sig`` ignored). POSIX: snapshot descendants via
    psutil; for SIGKILL, stop and rescan the live tree so a concurrent fork cannot
    escape a stale snapshot. Signal identity-checked descendants before their
    parent, then its group when ``pid`` leads one. Stopping is best-effort with a
    bounded wait; unavailable psutil still leaves process-group cleanup. Other
    signals do not suspend recipients. ``sig`` defaults to ``SIGKILL``."""
    if sys.platform == "win32":
        try:
            from runtime.subprocess_compat import windows_hide_flags
            creationflags = windows_hide_flags()
        except Exception:
            creationflags = 0
        try:
            proc = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=15, check=False, creationflags=creationflags,
            )
            # taskkill exits non-zero for not-found / access-denied (False = nothing terminated).
            return proc.returncode == 0
        except Exception:
            logger.debug("kill_process_tree: taskkill failed for pid %s", pid, exc_info=True)
            return False

    import signal as _signal
    if sig is None:
        sig = _signal.SIGKILL

    with _process_tree_snapshot(int(pid), hard_kill=sig == _signal.SIGKILL) as descendants:
        signalled = False
        # Signal descendants while their ownership ancestry is still observable.
        # Frozen hard-kill targets cannot fork during this bottom-up teardown.
        for child in reversed(descendants):
            try:
                if child.is_running():
                    child.send_signal(sig)
                    signalled = True
            except Exception:
                continue
        try:
            # getpgid→killpg has an inherent TOCTOU shared by every killpg site; the psutil
            # sweep below is identity-aware (PID + create time) and does not.
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError, OSError):
            pgid = None
        try:
            if pgid is not None and pgid == pid:
                # pid leads its own group (the == check avoids signalling the caller's group).
                os.killpg(pgid, sig)  # windows-footgun: ok — POSIX-only branch (win32 returns above)
            else:
                os.kill(pid, sig)
            signalled = True
        except ProcessLookupError:
            pass
        except (PermissionError, OSError):
            logger.debug("kill_process_tree: signal failed for pid %s", pid, exc_info=True)

        return signalled

def kill_popen_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort terminate a retained Popen and its descendants; never raises."""
    try:
        kill_process_tree(proc.pid)
    except Exception:
        logger.debug("kill_popen_process_tree: tree termination failed for pid %s", proc.pid, exc_info=True)
    # Ensure Popen's own bookkeeping sees the exit so communicate()/wait() cannot hang.
    try:
        proc.kill()
    except OSError:
        pass

__all__ = [
    "attach_self_to_kill_on_close_job",
    "kill_popen_process_tree",
    "kill_process_tree",
    "spawn_contained_process",
]
