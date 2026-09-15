"""Per-attempt process containment for Kanban workers on Windows.

A Kanban worker's death must not leave its descendants running. On POSIX a worker
is spawned with ``start_new_session=True``, so signalling its process group reaches
everything it started. That flag is ``setsid()`` — POSIX-only — so on Windows it is
a silent no-op, the worker has no group, and a kill reaches the worker PID and
nothing else. Measured consequence: a killed worker's PowerShell descendant kept
running for 288s and overlapped an entire 255s recovery attempt.

The Windows equivalent of a process group is a **Job Object**: the OS owns the
containment, so it holds even when the parent dies ungracefully.

Ownership model — the job handle lives in the WORKER, not the dispatcher
-----------------------------------------------------------------------
The job is created **named** and a handle to it is duplicated *into the worker
process* before the worker is resumed. The dispatcher then closes its own copy.

That is the whole design, and it is forced by two failures that a
dispatcher-owned handle cannot avoid:

* a **one-shot dispatcher** (``hermes kanban dispatch`` as a CLI) exits as soon as
  it has spawned. With ``KILL_ON_JOB_CLOSE`` a dispatcher-held handle would be the
  last one, so the dispatcher's exit would kill the worker it had just dispatched;
* a **long-lived gateway** dispatcher would accumulate one live handle per
  completed worker, leaking a handle per attempt forever.

Because the worker owns the handle, both problems disappear: the worker keeps its
own containment while it runs, and when the worker dies the kernel closes its
handle, which triggers ``KILL_ON_JOB_CLOSE`` and reaps the tree automatically.

The name lets a *different* process (a later dispatcher doing timeout/reclaim
cleanup) reopen the same job and terminate it explicitly.

Atomicity
---------
Containment is established before the worker can execute an instruction
(``CREATE_SUSPENDED`` → assign → duplicate → resume). If any step fails the child
is terminated, every handle is closed, and the failure is raised — a permanently
suspended or uncontained worker is never handed back as if it had spawned.

POSIX keeps its existing process-group behaviour and is untouched by any of this.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
from ctypes import wintypes
from typing import Any, Optional

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# CREATE_SUSPENDED: start the child before it can execute a single instruction, so
# nothing it spawns can precede job assignment.
_CREATE_SUSPENDED = 0x00000004

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
# Descendants of a worker are themselves job members, so they cannot break away.
# Deliberately NO breakaway flags: unlike the self-attach job (which must let
# gateway-relaunch children escape), this job's whole purpose is to hold a tree.
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1

# Access rights for OpenProcess / OpenJobObject.
_PROCESS_ALL_ACCESS = 0x1F0FFF  # includes PROCESS_DUP_HANDLE, needed to dupe into the worker
_JOB_OBJECT_TERMINATE = 0x0008
_JOB_OBJECT_QUERY = 0x0004

_DUPLICATE_SAME_ACCESS = 0x0002

#: Reopening a job by name is how a *different* dispatcher reaches an attempt's
#: tree. Bounded name so it cannot approach the 260-char object-name limit.
_JOB_NAME_PREFIX = "hermes-kanban-attempt-"
_JOB_NAME_MAX = 200


class WorkerSpawnError(RuntimeError):
    """Raised when a worker could not be started under containment."""


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
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


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    """Used to *prove* a job is empty instead of assuming termination worked."""

    _fields_ = [
        ("TotalUserTime", wintypes.LARGE_INTEGER),
        ("TotalKernelTime", wintypes.LARGE_INTEGER),
        ("ThisPeriodTotalUserTime", wintypes.LARGE_INTEGER),
        ("ThisPeriodTotalKernelTime", wintypes.LARGE_INTEGER),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", ctypes.c_long),
        ("tpDeltaPri", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
    ]


_kernel32 = None


def _k32():
    """``kernel32`` with explicit signatures.

    Every Win32 prototype is declared. Without ``restype``/``argtypes`` ctypes
    assumes a 32-bit ``int`` return, which silently truncates the 64-bit HANDLEs
    this module passes around — working by luck on some builds and corrupting the
    handle on others.
    """
    global _kernel32
    if _kernel32 is not None:
        return _kernel32
    if not _IS_WINDOWS:
        return None

    k = ctypes.WinDLL("kernel32", use_last_error=True)

    k.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    k.OpenJobObjectW.restype = wintypes.HANDLE
    k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    k.SetInformationJobObject.restype = wintypes.BOOL
    k.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)
    ]
    k.QueryInformationJobObject.restype = wintypes.BOOL
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k.AssignProcessToJobObject.restype = wintypes.BOOL
    k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k.TerminateJobObject.restype = wintypes.BOOL
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.DuplicateHandle.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD, wintypes.BOOL, wintypes.DWORD,
    ]
    k.DuplicateHandle.restype = wintypes.BOOL
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CloseHandle.restype = wintypes.BOOL
    k.GetCurrentProcess.argtypes = []
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetLastError.argtypes = []
    k.GetLastError.restype = wintypes.DWORD
    k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(_THREADENTRY32)]
    k.Thread32First.restype = wintypes.BOOL
    k.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_THREADENTRY32)]
    k.Thread32Next.restype = wintypes.BOOL
    k.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenThread.restype = wintypes.HANDLE
    k.ResumeThread.argtypes = [wintypes.HANDLE]
    k.ResumeThread.restype = wintypes.DWORD
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.WaitForSingleObject.restype = wintypes.DWORD

    _kernel32 = k
    return k


def windows_job_creationflags() -> int:
    """``CREATE_SUSPENDED`` on Windows so containment can be assigned before exec."""
    return _CREATE_SUSPENDED if _IS_WINDOWS else 0


def process_create_time(pid: Optional[int]) -> Optional[float]:
    """``(pid, create_time)`` identity for ``pid``; ``None`` when unavailable.

    PIDs are recycled, so a bare PID is not an identity. Hermes already uses
    ``(pid, create_time)`` for exactly this reason (``hermes_cli.process_identity``);
    the fallback kill path below is permitted only when the pair still matches.
    """
    if not pid:
        return None
    try:
        import psutil

        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        return None


def attempt_job_name(pid: int) -> str:
    """Stable, reopenable name for one attempt's job, derived from the worker PID.

    Derived rather than stored so that *any* process can reach an attempt's tree —
    a timeout sweep, a reclaim, or a crash-detection pass running in a different
    dispatcher — with nothing but the ``worker_pid`` already persisted on the task.

    PID reuse is safe here: the name is held only for as long as some handle to the
    job is open, and after handoff the worker is the sole holder. When the worker
    dies its handle closes, the kernel destroys the job and frees the name, so a
    later worker reusing that PID creates a genuinely new job rather than joining a
    stale one.
    """
    return ("%s%d" % (_JOB_NAME_PREFIX, int(pid)))[:_JOB_NAME_MAX]


def resume_worker(pid: int) -> bool:
    """Resume every thread of a worker spawned with ``CREATE_SUSPENDED``.

    No-op returning ``False`` off Windows. Never raises.
    """
    k = _k32()
    if k is None:
        return False
    try:
        TH32CS_SNAPTHREAD = 0x00000004
        THREAD_SUSPEND_RESUME = 0x0002
        snapshot = k.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        if not snapshot or snapshot == wintypes.HANDLE(-1).value:
            return False
        try:
            entry = _THREADENTRY32()
            entry.dwSize = ctypes.sizeof(_THREADENTRY32)
            ok = k.Thread32First(snapshot, ctypes.byref(entry))
            resumed = False
            while ok:
                if entry.th32OwnerProcessID == int(pid):
                    thread = k.OpenThread(THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                    if thread:
                        try:
                            # ResumeThread returns (DWORD)-1 on failure.
                            if k.ResumeThread(thread) != 0xFFFFFFFF:
                                resumed = True
                        finally:
                            k.CloseHandle(thread)
                ok = k.Thread32Next(snapshot, ctypes.byref(entry))
            return resumed
        finally:
            k.CloseHandle(snapshot)
    except Exception:
        logger.debug("resume failed for pid %s", pid, exc_info=True)
        return False


def _terminate_spawned_child(process: Any, pid: int) -> None:
    """Terminate the exact child we just spawned, before it has executed.

    Every failure path that reaches here leaves the child **unresumed**, so it
    cannot have spawned descendants: ending that one process is complete cleanup
    and a tree walk would be pointless.

    The ``Popen`` object is preferred because it addresses the exact process we
    created rather than looking a PID up, so there is no reuse window to reason
    about. ``taskkill /T`` and the named-job reopen belong to later lifecycle
    cleanup, after a worker has actually run.
    """
    if process is not None:
        try:
            process.kill()
            try:
                process.wait(timeout=10)
            except Exception:
                logger.debug("spawned child %s did not reap within timeout", pid)
            return
        except Exception:
            logger.debug("could not kill spawned child %s directly", pid, exc_info=True)
    if not pid:
        return
    # No Popen object: while we hold this child unreaped its PID cannot have been
    # recycled, so a direct kill of that exact PID (no /T) is still safe here.
    try:
        from hermes_cli._subprocess_compat import windows_hide_flags

        flags = windows_hide_flags()
    except Exception:
        flags = 0
    try:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(int(pid))],
            capture_output=True, timeout=10, check=False, creationflags=flags,
        )
    except Exception:
        logger.debug("taskkill /F /PID %s failed", pid, exc_info=True)


def contain_and_resume(pid: int, job_name: str, *, process: Any = None) -> dict[str, Any]:
    """Atomically place a suspended worker under a named kill-on-close job.

    Sequence (every step's result is checked)::

        CreateJobObjectW(name) -> SetInformationJobObject(KILL_ON_JOB_CLOSE)
          -> OpenProcess(worker) -> AssignProcessToJobObject
          -> DuplicateHandle(job -> worker)   # worker owns containment
          -> ResumeThread
          -> CloseHandle(job)                 # dispatcher drops its copy

    The duplicate-into-worker step is what stops a one-shot dispatcher's exit from
    killing the worker it just spawned, and what stops a long-lived dispatcher from
    leaking a handle per completed worker.

    On success returns the containment metadata to persist on the run. On any
    failure the worker is terminated, every handle is closed, and
    :class:`WorkerSpawnError` is raised — the caller must not receive a PID for a
    worker that is not running under containment.
    """
    if not _IS_WINDOWS:
        return {"contained": False, "job_name": None, "reason": "not_windows"}

    k = _k32()
    job = None
    process_handle = None
    duplicated = wintypes.HANDLE()
    try:
        job = k.CreateJobObjectW(None, job_name)
        if not job:
            raise WorkerSpawnError("CreateJobObjectW failed (err=%s)" % k.GetLastError())

        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(
            job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise WorkerSpawnError("SetInformationJobObject failed (err=%s)" % k.GetLastError())

        # OpenProcess races the child's own startup, so retry briefly rather than
        # silently dropping containment.
        import time as _time

        for _ in range(20):
            process_handle = k.OpenProcess(_PROCESS_ALL_ACCESS, False, int(pid))
            if process_handle:
                break
            _time.sleep(0.05)
        if not process_handle:
            # Fail closed. A worker we cannot open is a worker whose process tree
            # we cannot own, and Kanban must never record an executor it cannot
            # contain — that ambiguity is exactly what the qualification exposed.
            # A host whose job layout forbids per-attempt containment does not
            # satisfy the executor-containment contract; failing the spawn is the
            # correct safe behaviour. Hermes' own Windows self-job already sets
            # BREAKAWAY_OK|SILENT_BREAKAWAY_OK, so the normal environment permits
            # children to escape into their own job.
            raise WorkerSpawnError(
                "OpenProcess failed for pid %s (err=%s)" % (pid, k.GetLastError())
            )

        if not k.AssignProcessToJobObject(job, process_handle):
            raise WorkerSpawnError("AssignProcessToJobObject refused pid %s (err=%s)" % (pid, k.GetLastError()))

        if not k.DuplicateHandle(
            k.GetCurrentProcess(), job, process_handle,
            ctypes.byref(duplicated), 0, False, _DUPLICATE_SAME_ACCESS,
        ):
            raise WorkerSpawnError("DuplicateHandle into pid %s failed (err=%s)" % (pid, k.GetLastError()))

        if not resume_worker(pid):
            raise WorkerSpawnError("could not resume worker pid %s" % pid)

        return {
            "contained": True,
            "job_name": job_name,
            "worker_pid": int(pid),
            "worker_create_time": process_create_time(pid),
            "reason": None,
        }
    except Exception:
        # Fail closed: never hand back a PID for a suspended/uncontained worker.
        # The child has never been resumed, so terminating that exact process
        # object is complete cleanup — it cannot have spawned descendants.
        _terminate_spawned_child(process, int(pid))
        raise
    finally:
        if duplicated:
            k.CloseHandle(duplicated)
        if process_handle:
            k.CloseHandle(process_handle)
        if job:
            k.CloseHandle(job)


def _job_active_processes(job_handle) -> Optional[int]:
    """Active process count for an open job handle, or ``None`` if unqueryable."""
    k = _k32()
    if k is None:
        return None
    try:
        acct = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        returned = wintypes.DWORD()
        if k.QueryInformationJobObject(
            job_handle, _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(acct), ctypes.sizeof(acct), ctypes.byref(returned),
        ):
            return int(acct.ActiveProcesses)
        return None
    except Exception:
        return None


def terminate_worker_job(pid: Optional[int], job_name: Optional[str] = None) -> dict[str, Any]:
    """End an attempt's contained tree and *prove* it is empty.

    Works from a different process than the one that spawned the worker: the job is
    reopened by name, so timeout/reclaim cleanup does not need the dispatcher's
    in-memory handle table.

    Returns the cleanup disposition recorded on the run:
    ``contained``, ``tree_terminated``, ``survivors`` (``None`` = not proven).
    """
    result: dict[str, Any] = {"contained": False, "tree_terminated": False, "survivors": None}
    k = _k32()
    if k is None or not job_name:
        return result

    job = k.OpenJobObjectW(_JOB_OBJECT_TERMINATE | _JOB_OBJECT_QUERY, False, job_name)
    if not job:
        logger.debug("could not reopen job %s for pid %s", job_name, pid)
        return result
    result["contained"] = True
    try:
        if not k.TerminateJobObject(job, 1):
            logger.debug("TerminateJobObject failed for %s (err=%s)", job_name, k.GetLastError())
            return result
        # Termination is asynchronous; prove emptiness rather than assume it.
        import time as _time

        survivors = None
        for _ in range(50):
            survivors = _job_active_processes(job)
            if survivors == 0:
                break
            _time.sleep(0.1)
        result["tree_terminated"] = survivors == 0
        result["survivors"] = survivors
        return result
    except Exception:
        logger.debug("job termination failed for %s", job_name, exc_info=True)
        return result
    finally:
        k.CloseHandle(job)


class _Unguarded:
    """Sentinel: the caller asserts this PID is the worker it just created.

    Used only on the spawn failure path, where the child is still suspended and
    cannot have been recycled. Every PID-addressed kill that could race a recycled
    PID must pass a real ``(pid, create_time)`` identity instead.
    """

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "<unguarded>"


_UNGUARDED = _Unguarded()


def terminate_pid_tree(pid: int, expected_create_time: Any = _UNGUARDED) -> bool:
    """Fallback whole-tree kill for a worker with no containment job.

    Used only when the job could not be created or reopened (e.g. a worker spawned
    by an older build). Windows: ``taskkill /F /T``.

    ``expected_create_time`` — the ``create_time`` recorded for this worker when it
    was spawned. PIDs are recycled, so a bare PID is not an identity:

    * a **float** — the kill proceeds only if the live process still has that
      ``create_time``; a mismatch means the PID now belongs to something else and
      the kill is refused;
    * ``None`` — identity is **not proven** (no spawn record); refused;
    * omitted — unguarded, for the caller that just created the process.
    """
    if not _IS_WINDOWS or not pid:
        return False
    if expected_create_time is not _UNGUARDED:
        if expected_create_time is None:
            logger.warning("refusing tree kill of pid %s: no recorded spawn identity", pid)
            return False
        actual = process_create_time(pid)
        if actual is None or abs(actual - float(expected_create_time)) > 0.001:
            logger.warning(
                "refusing tree kill of pid %s: create_time %s != expected %s (PID reused?)",
                pid, actual, expected_create_time,
            )
            return False
    try:
        proc = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(int(pid))],
            capture_output=True, timeout=15, check=False, creationflags=0x08000000,
        )
        return proc.returncode == 0
    except Exception:
        logger.debug("taskkill /F /T /PID %s failed", pid, exc_info=True)
        return False
