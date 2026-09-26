"""Contain each managed Windows router tree without adopting the owner process."""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
from ctypes import wintypes
import logging
import subprocess
import sys
import threading

import psutil

logger = logging.getLogger(__name__)


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
    def __init__(self):
        self._lock = threading.Lock()
        self._api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int,
                                         ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
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
                    self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
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


_CREDENTIAL_ENV_MARKERS = ("_API_KEY", "_TOKEN", "_SECRET", "PASSWORD", "_CREDENTIALS")


def server_child_env(base_env: Mapping[str, str]) -> dict[str, str]:
    """Return the environment a native inference child (llama-server) gets.

    Provider and tool credentials never belong in a native child that talks to nobody but
    us — and on Windows they are not merely leaked: the bundled OpenMP runtime died with
    STATUS_HEAP_CORRUPTION during initialisation with one `*_API_KEY` present and loaded fine
    with only that variable removed (#116109, confirmed on a model-free libomp.dll probe).
    Everything else (PATH, CUDA_*, HSA_*, OMP_*, TEMP, …) passes through untouched. Applied by
    the llama-server supervisor only: spawn_server is also the generic bounded-probe spawner
    (git / PowerShell / update probes), whose children legitimately need GH_TOKEN, HF_TOKEN, …
    """
    return {
        key: value for key, value in base_env.items()
        if not any(marker in key.upper() for marker in _CREDENTIAL_ENV_MARKERS)
    }


def spawn_server(cmd, **kwargs) -> tuple[subprocess.Popen, _WindowsJob | None]:
    """Start a router, returning its process and an owner-held containment handle.

    Keep the job until shutdown and call close() to terminate the entire tree.
    Windows closes it automatically if the owner dies. Other hosts retain Popen's
    ordinary behavior. Assignment happens before the child's first instruction.
    """
    if sys.platform != "win32":
        return subprocess.Popen(cmd, **kwargs), None
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


_kill_on_exit_lock = threading.Lock()
_kill_on_exit_job: _WindowsJob | None = None
_kill_on_exit_failed = False
_assign_warned = False
_session_assign_warned = False
_assign_warn_lock = threading.Lock()


def kill_on_exit_job() -> _WindowsJob | None:
    """Process-lifetime containment job. Never closed.

    None off Windows. A creation failure logs one warning and is not retried.
    """
    global _kill_on_exit_job, _kill_on_exit_failed
    if sys.platform != "win32":
        return None
    if _kill_on_exit_job is not None or _kill_on_exit_failed:
        return _kill_on_exit_job
    with _kill_on_exit_lock:
        if _kill_on_exit_job is not None or _kill_on_exit_failed:
            return _kill_on_exit_job
        try:
            _kill_on_exit_job = _WindowsJob()
        except OSError as exc:
            _kill_on_exit_failed = True
            logger.warning("kill-on-exit job creation failed: %s", exc)
            return None
        return _kill_on_exit_job


def _warn_assign_once(exc: BaseException, *, session: bool = False) -> None:
    global _assign_warned, _session_assign_warned
    with _assign_warn_lock:
        if session:
            if _session_assign_warned:
                return
            _session_assign_warned = True
        else:
            if _assign_warned:
                return
            _assign_warned = True
    kind = "session job" if session else "kill-on-exit job"
    logger.warning("%s assign failed: %s", kind, exc)


def spawn_contained(cmd, *, job, session: bool = False, **kwargs) -> subprocess.Popen:
    """Start ``cmd``. With a Windows job: suspend, assign, resume.

    Assign failure logs once and still resumes. A warning-log failure does
    not skip that resume. ``session=True`` logs that warning as a session
    job, once, separate from the process-lifetime job. ``psutil.NoSuchProcess``,
    or any other resume error when ``proc`` has no real process handle, is
    not a suspended child: return it and do not kill. Any other escape after
    ``Popen`` before that return — including ``KeyboardInterrupt`` during
    assign or warn, and resume failure of a real process handle — kills
    that child and re-raises. This never closes ``job``.
    Non-Windows, or ``job is None``, is a plain Popen.
    """
    if sys.platform != "win32" or job is None:
        return subprocess.Popen(cmd, **kwargs)
    kwargs["creationflags"] = kwargs.get("creationflags", 0) | 0x00000004  # CREATE_SUSPENDED
    proc = subprocess.Popen(cmd, **kwargs)
    try:
        try:
            job.assign(proc)
        except Exception as exc:
            try:
                _warn_assign_once(exc, session=session)
            except Exception:
                pass
        try:
            psutil.Process(proc.pid).resume()
        except psutil.NoSuchProcess:
            return proc
        except BaseException as exc:
            handle = getattr(proc, "_handle", None)
            if not isinstance(handle, int) or isinstance(handle, bool):
                if not isinstance(exc, Exception):
                    raise
                return proc
            raise
        return proc
    except BaseException:
        handle = getattr(proc, "_handle", None)
        if isinstance(handle, int) and not isinstance(handle, bool):
            try:
                proc.kill()
                proc.wait(timeout=10)
            finally:
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
                proc._handle.Close()
        raise
