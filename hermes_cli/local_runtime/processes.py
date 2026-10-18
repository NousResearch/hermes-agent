"""Contain each managed Windows router tree without adopting the owner process."""

from __future__ import annotations

from collections.abc import Mapping
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
_ERROR_MORE_DATA = 234
_WAIT_TIMEOUT = 0x102


class _WindowsJob:
    def __init__(self):
        self._lock = threading.Lock()
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

    def terminate(self, timeout: float = 5.0) -> bool:
        """Kill the contained tree, wait until every member has exited, then close the job.

        Termination is asynchronous, so close() alone returns while members may still run, and
        the job's ActiveProcesses count was seen at zero while a killed member still existed.
        Each member is therefore opened before the kill and its own handle waited on; members
        started while the tree was dying are caught by listing the job again until it is empty.
        False if members remained at the deadline (the handle is closed either way); an
        already closed job counts as empty.
        """
        try:
            with self._lock:
                if self._handle is None:
                    return True
                deadline = time.monotonic() + timeout
                while True:
                    pids = self._member_pids()
                    if not pids:
                        return True
                    handles = self._open_members(pids)
                    try:
                        if not self._api.TerminateJobObject(self._handle, 1):
                            raise ctypes.WinError(ctypes.get_last_error())
                        for handle in handles:
                            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                            waited = self._api.WaitForSingleObject(handle, remaining_ms)
                            if waited == _WAIT_TIMEOUT:
                                return False
                            if waited != 0:  # WAIT_OBJECT_0
                                raise ctypes.WinError(ctypes.get_last_error())
                    finally:
                        for handle in handles:
                            self._api.CloseHandle(handle)
                    if time.monotonic() >= deadline:
                        return False
                    time.sleep(0.01)
        finally:
            self.close()

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

    def _open_members(self, pids: list[int]) -> list[int]:
        """Open each listed member. The handle pins the PID, and the membership check made
        through it rejects a PID recycled between the listing and the open. A member that could
        not be opened is still listed on the next pass, so it is never taken for gone."""
        handles = []
        try:
            for pid in pids:
                handle = self._api.OpenProcess(_SYNCHRONIZE_AND_QUERY, False, pid)
                if not handle:
                    continue
                in_job = wintypes.BOOL()
                if (self._api.IsProcessInJob(handle, self._handle, ctypes.byref(in_job))
                        and in_job.value):
                    handles.append(handle)
                else:
                    self._api.CloseHandle(handle)
        except BaseException:
            for handle in handles:
                self._api.CloseHandle(handle)
            raise
        return handles

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
