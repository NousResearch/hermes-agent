"""Contain each managed router tree without adopting the owner process."""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
from ctypes import wintypes
import os
import subprocess
import sys
import threading

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


# The POSIX counterpart of the Windows job. The idle sweeper and the supervisor's
# watchdog thread both live INSIDE the owner, so on an owner crash nothing that
# could unload the model survives — the router and its multi-GB model children
# get reparented to launchd/systemd and stay resident (#126631). A reaper must
# therefore be a separate process: a tiny sibling that polls the owner identity
# (PID + create time, so a reused owner PID can never keep a dead owner "alive"),
# then terminates the same tree KILL_ON_JOB_CLOSE would have taken down.
_POSIX_REAPER_SCRIPT = r'''
"""hermes-owner-death-reaper: terminate a router tree whose owner has exited."""
import sys
import time
from contextlib import suppress

import psutil

owner_pid, owner_created, router_pid, router_created = (
    int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))


def still_running(pid, created):
    """True only for the exact recorded incarnation — a reused PID is a stranger."""
    try:
        proc = psutil.Process(pid)
        # A zombie holds no memory and needs nothing from us: count it as gone
        # so an unreaped corpse cannot keep the reaper polling forever.
        return (proc.create_time() == created and proc.is_running()
                and proc.status() != psutil.STATUS_ZOMBIE)
    except (psutil.Error, ValueError, OverflowError, OSError):
        return False


def owner_alive():
    try:
        owner = psutil.Process(owner_pid)
        # A newer incarnation proves the recorded owner exited (PID reuse).
        # A zombie owner is equally gone — the same unreaped corpse must not
        # keep this reaper polling forever (see still_running above).
        return (owner.create_time() == owner_created and owner.is_running()
                and owner.status() != psutil.STATUS_ZOMBIE)
    except (psutil.Error, ValueError, OverflowError, OSError):
        return False


def terminate_tree():
    children = []
    with suppress(psutil.Error):
        children = psutil.Process(router_pid).children(recursive=True)
    victims = children
    if still_running(router_pid, router_created):
        victims.append(psutil.Process(router_pid))
    for proc in victims:
        with suppress(psutil.Error):
            proc.terminate()
    # A terminated non-child can only become a zombie here — the OS reaps it on
    # its own schedule, so waiting it out would just burn the full timeout.
    # Give SIGTERM a short grace, escalate to SIGKILL, and let the OS collect.
    deadline = time.monotonic() + 3
    for proc in victims:
        with suppress(psutil.Error):
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except psutil.TimeoutExpired:
                proc.kill()


while True:
    if not still_running(router_pid, router_created):
        raise SystemExit(0)  # Router exited via stop() or crashed: nothing to reap.
    if not owner_alive():
        terminate_tree()
        raise SystemExit(0)
    time.sleep(2)
'''


def _spawn_owner_death_reaper(proc: subprocess.Popen) -> None:
    """Leave a sibling that reaps the router tree when this process dies.

    The reaper exits by itself once the router is gone, so a graceful stop()
    leaves no residue beyond one polling cycle. A daemon thread waits it out to
    keep the exited reaper from piling up as a zombie while the owner lives on.
    """
    try:
        router_created = psutil.Process(proc.pid).create_time()
    except psutil.Error:
        return  # Already gone: the tree died with it, nothing left to reap.
    cmd = [
        sys.executable, "-c", _POSIX_REAPER_SCRIPT,
        str(os.getpid()), str(psutil.Process().create_time()),
        str(proc.pid), str(router_created),
    ]
    reaper = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    threading.Thread(target=reaper.wait, daemon=True,
                     name="llamacpp-owner-reaper-wait").start()


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


def spawn_server(cmd, *, reap_with_owner: bool = False, **kwargs) -> tuple[subprocess.Popen, _WindowsJob | None]:
    """Start a router, returning its process and an owner-held containment handle.

    Keep the job until shutdown and call close() to terminate the entire tree.
    Windows closes it automatically if the owner dies. On other hosts the caller
    passes reap_with_owner=True to arm a sibling reaper doing the same; without
    it Popen's ordinary behavior applies (this is also the generic bounded-probe
    spawner, whose short-lived children need no supervision). Assignment happens
    before the child's first instruction.
    """
    if sys.platform != "win32":
        proc = subprocess.Popen(cmd, **kwargs)
        if reap_with_owner:
            _spawn_owner_death_reaper(proc)
        return proc, None
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
