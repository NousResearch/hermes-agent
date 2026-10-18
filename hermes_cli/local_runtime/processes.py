"""Contain each managed Windows router tree without adopting the owner process."""

from __future__ import annotations

from collections.abc import Mapping
import contextlib
import ctypes
from ctypes import wintypes
import subprocess
import sys
import threading
import time

import psutil


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


def _process_id_list(capacity: int) -> ctypes.Structure:
    class _ProcessIdList(ctypes.Structure):
        _fields_ = [
            ("NumberOfAssignedProcesses", wintypes.DWORD),
            ("NumberOfProcessIdsInList", wintypes.DWORD),
            ("ProcessIdList", ctypes.c_size_t * capacity),
        ]
    return _ProcessIdList()


_SYNCHRONIZE_AND_QUERY = 0x00100000 | 0x1000  # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
_ERROR_INVALID_PARAMETER = 87
_ERROR_MORE_DATA = 234
_WAIT_TIMEOUT = 0x102


def _remaining_ms(deadline: float) -> int:
    return max(0, int((deadline - time.monotonic()) * 1000))


class JobShutdownError(RuntimeError):
    """The contained processes were not all confirmed exited; the job was left open."""


class _WindowsJob:
    def __init__(self):
        self._lock = threading.Lock()
        self._tracked: list[subprocess.Popen] = []
        self._members: dict[int, int] = {}  # pid -> held SYNCHRONIZE handle, until it signals
        self._unconfirmed: set[int] = set()  # listed pids not yet opened nor proven gone
        self._api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int,
                                         ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            ("QueryInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                           wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            ("IsProcessInJob", [wintypes.HANDLE, wintypes.HANDLE,
                                ctypes.POINTER(wintypes.BOOL)], wintypes.BOOL),
            ("WaitForSingleObject", [wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
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

    def track(self, proc: subprocess.Popen) -> None:
        """Take over a process whose start failed. Its assignment may have failed too, leaving
        it outside the job, so terminate() kills it and confirms its exit through its Popen
        handle like a member's; its pipes and handle are released when the job closes."""
        self._tracked.append(proc)

    def terminate(self, timeout: float) -> None:
        """Kill the contained tree and return once every member has exited; only then close.

        Termination is asynchronous, so close() alone returns while members may still run, and
        the job's ActiveProcesses count was seen at zero while a killed member still existed.
        Each member is therefore opened before the kill and its own handle waited on; members
        started while the tree was dying are caught by listing the job again until it is empty.
        A member's handle is kept until it signals, across passes and calls, so one that left
        the listing without being seen to exit is still awaited; a listed PID that could not be
        opened stays unconfirmed until a later open settles it (see _open_members), never merely
        because the listing stopped showing it. If the tree is not confirmed
        gone by the deadline, or a WinAPI call fails, JobShutdownError is raised and the job
        stays open: KILL_ON_JOB_CLOSE stays armed and the members held, so a later terminate()
        can still confirm them. An already closed job counts as empty.
        """
        with self._lock:
            if self._handle is None:
                return
            deadline = time.monotonic() + timeout
            try:
                while not self._end_members(deadline):
                    time.sleep(0.01)
                self._close_locked()
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise JobShutdownError("contained processes could not be confirmed exited") from exc

    def _end_members(self, deadline: float) -> bool:
        """One pass: kill every member and wait for each held one. True once none is left."""
        for proc in self._tracked:
            with contextlib.suppress(OSError):  # the wait below decides whether it exited
                proc.kill()
        pids = self._member_pids()
        self._open_members(pids)
        # A listed member that could not be opened keeps the job unconfirmed.
        if (not pids and not self._members and not self._unconfirmed
                and all(p.poll() is not None for p in self._tracked)):
            return True
        if time.monotonic() >= deadline:
            raise JobShutdownError("contained processes were still running at the deadline")
        if not self._api.TerminateJobObject(self._handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        for pid, handle in list(self._members.items()):
            waited = self._api.WaitForSingleObject(handle, _remaining_ms(deadline))
            if waited == _WAIT_TIMEOUT:
                raise JobShutdownError("a contained process was still running at the deadline")
            if waited != 0:  # WAIT_OBJECT_0
                raise ctypes.WinError(ctypes.get_last_error())
            del self._members[pid]
            self._api.CloseHandle(handle)
        for proc in self._tracked:
            proc.wait(timeout=_remaining_ms(deadline) / 1000)
        return False

    def _member_pids(self) -> list[int]:
        capacity = 16
        while True:
            info = _process_id_list(capacity)
            if self._api.QueryInformationJobObject(  # 3 = BasicProcessIdList
                    self._handle, 3, ctypes.byref(info), ctypes.sizeof(info), None):
                if info.NumberOfProcessIdsInList >= info.NumberOfAssignedProcesses:
                    return list(info.ProcessIdList[:info.NumberOfProcessIdsInList])
            elif (error := ctypes.get_last_error()) != _ERROR_MORE_DATA:
                raise ctypes.WinError(error)
            capacity = max(capacity * 2, info.NumberOfAssignedProcesses)

    def _open_members(self, pids: list[int]) -> None:
        """Hold each listed member not held yet. The handle pins the PID (a listed PID that is
        held is that same process), and the membership check made through it rejects a PID
        recycled between the listing and the open.

        A listed PID is let go unheld only on evidence that its process is gone: no process has
        that PID (ERROR_INVALID_PARAMETER; a process object outlives its exit while any handle
        is open, and a PID is reused only once it is deleted), or the PID now names a process
        outside the job (members cannot break away, so it was recycled). Any other failure
        leaves it unconfirmed and retried on every pass: the listing drops a member while it is
        still being torn down, so its absence from a later listing is no proof it exited."""
        for pid in set(pids) | self._unconfirmed:
            if pid in self._members:
                continue
            handle = self._api.OpenProcess(_SYNCHRONIZE_AND_QUERY, False, pid)
            if not handle:
                if ctypes.get_last_error() == _ERROR_INVALID_PARAMETER:
                    self._unconfirmed.discard(pid)
                else:
                    self._unconfirmed.add(pid)
                continue
            in_job = wintypes.BOOL()
            if not self._api.IsProcessInJob(handle, self._handle, ctypes.byref(in_job)):
                self._unconfirmed.add(pid)
                self._api.CloseHandle(handle)
                continue
            self._unconfirmed.discard(pid)
            if in_job.value:
                self._members[pid] = handle
            else:
                self._api.CloseHandle(handle)

    def close(self) -> None:
        """Terminate the contained tree without waiting; repeated closes are harmless."""
        with self._lock:
            for proc in self._tracked:
                with contextlib.suppress(OSError):
                    proc.kill()
            self._close_locked()

    def _close_locked(self) -> None:
        for proc in self._tracked:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
            proc._handle.Close()
        self._tracked.clear()
        for handle in self._members.values():
            self._api.CloseHandle(handle)
        self._members.clear()
        self._unconfirmed.clear()
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
    try:
        return spawn_in_job(job, cmd, **kwargs), job
    except BaseException:
        try:
            job.terminate(timeout=10)
        finally:
            job.close()
        raise


def new_job() -> _WindowsJob:
    """An empty kill-on-close job for spawn_in_job (Windows only)."""
    return _WindowsJob()


def spawn_in_job(job: _WindowsJob, cmd, **kwargs) -> subprocess.Popen:
    """Start a process inside a caller-owned job, assigned before its first instruction.

    If the start fails the job takes the process over (track()), so the caller's terminate()
    confirms it gone along with anything it started, whether or not it was assigned.
    """
    proc = None
    try:
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | 0x00000004  # CREATE_SUSPENDED
        proc = subprocess.Popen(cmd, **kwargs)
        job.assign(proc)
        psutil.Process(proc.pid).resume()
        return proc
    except BaseException:
        if proc is not None:
            job.track(proc)
        raise
