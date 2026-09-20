"""Owned process-tree lifecycle for dispatcher-spawned Kanban workers.

The dispatcher must own the *whole* worker tree, not just the PID returned by
``Popen``.  Windows uses a named Job Object (with kill-on-close) and POSIX uses
a fresh process group.  Every destructive operation is scoped by a persisted
ownership id and a process creation-time fingerprint; a recycled PID is never
signalled.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a runtime dependency
    psutil = None  # type: ignore[assignment]


_IS_WINDOWS = sys.platform == "win32"
_DEFAULT_GRACE_SECONDS = 2.0
_DEFAULT_FORCE_WAIT_SECONDS = 3.0
_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class WorkerReceipt:
    """Stable identity and ownership evidence for one spawned worker tree."""

    pid: int
    start_time: int
    owner_kind: str
    owner_id: str
    envelope_path: Optional[str] = None
    command: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "start_time": self.start_time,
            "owner_kind": self.owner_kind,
            "owner_id": self.owner_id,
            "envelope_path": self.envelope_path,
            "command": list(self.command),
        }


@dataclass
class _WorkerRecord:
    receipt: WorkerReceipt
    process: subprocess.Popen
    owner: Any = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    root_exited: bool = False
    exit_code: Optional[int] = None
    known_descendants: dict[int, int] = field(default_factory=dict)


_RECORDS: dict[int, _WorkerRecord] = {}
# ``spawn_worker_process`` returns a Popen-compatible object but the existing
# Kanban dispatcher stores only its PID. Keep the receipt briefly after the
# monitor thread closes a fast-exiting process so ``_set_worker_pid`` can still
# persist ownership evidence without racing normal completion.
_RECENT_RECEIPTS: dict[int, tuple[WorkerReceipt, float]] = {}
_RECEIPT_RETENTION_SECONDS = 3600.0
_RECORDS_LOCK = threading.RLock()


def _process_start_time(pid: int) -> Optional[int]:
    """Return a stable microsecond process-creation fingerprint."""
    if psutil is None:
        return None
    try:
        return int(round(psutil.Process(int(pid)).create_time() * 1_000_000))
    except Exception:
        return None


def _wait_for_start_time(pid: int, timeout: float = 1.0) -> Optional[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = _process_start_time(pid)
        if value is not None:
            return value
        time.sleep(_POLL_SECONDS)
    return _process_start_time(pid)


def _pid_alive(pid: int) -> bool:
    if not pid or pid <= 0 or psutil is None:
        return False
    try:
        proc = psutil.Process(int(pid))
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _identity_state(pid: int, expected_start_time: Optional[int]) -> tuple[bool, bool]:
    """Return ``(is_alive, identity_matches)`` for a PID/start pair."""
    if not pid or pid <= 0 or psutil is None:
        return False, False
    try:
        proc = psutil.Process(int(pid))
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False, False
        if expected_start_time is None:
            return True, False
        actual = int(round(proc.create_time() * 1_000_000))
        return True, actual == int(expected_start_time)
    except Exception:
        return False, False


def _root_identity_definitely_absent(
    pid: int,
    expected_start_time: Optional[int],
) -> bool:
    """Return True only when the persisted root is provably gone.

    A missing Job Object handle normally means its kill-on-close object was
    already destroyed.  Before treating that as an absent tree, require a
    persisted creation fingerprint and a process-table probe that is
    unambiguously ``NoSuchProcess``/zombie.  Access failures and every live
    PID, including a recycled PID, remain ambiguous and therefore fail
    closed.
    """
    if not pid or pid <= 0 or expected_start_time is None or psutil is None:
        return False
    try:
        proc = psutil.Process(int(pid))
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return True
        # Read the creation time as part of the exact-identity probe.  A
        # mismatch is a live recycled PID, not proof that the old tree is
        # absent; an exception is an ambiguous probe, not proof either.
        int(round(proc.create_time() * 1_000_000))
        return False
    except psutil.NoSuchProcess:
        return True
    except Exception:
        return False


def _snapshot_descendants(pid: int) -> dict[int, int]:
    if psutil is None or not pid or pid <= 0:
        return {}
    try:
        proc = psutil.Process(int(pid))
        children = proc.children(recursive=True)
    except Exception:
        return {}
    result: dict[int, int] = {}
    for child in children:
        try:
            if child.status() == psutil.STATUS_ZOMBIE:
                continue
            result[int(child.pid)] = int(round(child.create_time() * 1_000_000))
        except Exception:
            continue
    return result


def _group_alive(pgid: int) -> bool:
    """Return whether a non-zombie process remains in a POSIX process group."""
    if not pgid or pgid <= 0:
        return False
    if psutil is not None:
        try:
            for proc in psutil.process_iter(["pid", "status"]):
                try:
                    if proc.info.get("status") == psutil.STATUS_ZOMBIE:
                        continue
                    if os.getpgid(int(proc.info["pid"])) == int(pgid):
                        return True
                except (OSError, ProcessLookupError, psutil.Error):
                    continue
            return False
        except Exception:
            pass
    try:
        os.killpg(int(pgid), 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, target)


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _update_envelope(path: Optional[str], **updates: Any) -> None:
    if not path:
        return
    payload = _read_json(path)
    payload.update(updates)
    try:
        _atomic_write_json(path, payload)
    except OSError:
        # Evidence must never take down the dispatcher. The DB event remains
        # authoritative when an operator has removed the sidecar directory.
        return


if _IS_WINDOWS:
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_QUERY = 0x0004
    _JOB_OBJECT_SET_ATTRIBUTES = 0x0002
    _JOB_OBJECT_TERMINATE = 0x0008
    _CREATE_NEW_PROCESS_GROUP = 0x00000200
    _CREATE_NO_WINDOW = 0x08000000

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", ctypes.wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.wintypes.DWORD),
            ("SchedulingClass", ctypes.wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _BasicAccountingInformation(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong),
            ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", ctypes.wintypes.DWORD),
            ("TotalProcesses", ctypes.wintypes.DWORD),
            ("ActiveProcesses", ctypes.wintypes.DWORD),
            ("TotalTerminatedProcesses", ctypes.wintypes.DWORD),
        ]

    class _WindowsJob:
        def __init__(self, handle: int, name: str):
            self.handle = handle
            self.name = name

        @staticmethod
        def _kernel32():
            kernel32 = ctypes.windll.kernel32
            kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.wintypes.LPCWSTR]
            kernel32.CreateJobObjectW.restype = ctypes.wintypes.HANDLE
            kernel32.OpenJobObjectW.argtypes = [ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.LPCWSTR]
            kernel32.OpenJobObjectW.restype = ctypes.wintypes.HANDLE
            kernel32.SetInformationJobObject.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.INT, ctypes.c_void_p, ctypes.wintypes.DWORD]
            kernel32.SetInformationJobObject.restype = ctypes.wintypes.BOOL
            kernel32.AssignProcessToJobObject.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.HANDLE]
            kernel32.AssignProcessToJobObject.restype = ctypes.wintypes.BOOL
            kernel32.TerminateJobObject.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.UINT]
            kernel32.TerminateJobObject.restype = ctypes.wintypes.BOOL
            kernel32.QueryInformationJobObject.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.INT, ctypes.c_void_p, ctypes.wintypes.DWORD, ctypes.POINTER(ctypes.wintypes.DWORD)]
            kernel32.QueryInformationJobObject.restype = ctypes.wintypes.BOOL
            kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
            kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
            return kernel32

        @classmethod
        def create(cls, *, kill_on_close: bool = True) -> "_WindowsJob":
            kernel32 = cls._kernel32()
            name = f"Local\\HermesKanbanWorker_{uuid.uuid4().hex}"
            handle = kernel32.CreateJobObjectW(None, name)
            if not handle:
                raise ctypes.WinError()
            job = cls(int(handle), name)
            if kill_on_close:
                info = _ExtendedLimitInformation()
                info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                ok = kernel32.SetInformationJobObject(
                    handle, 9, ctypes.byref(info), ctypes.sizeof(info)
                )
                if not ok:
                    job.close()
                    raise ctypes.WinError()
            return job

        @classmethod
        def open(cls, name: str) -> Optional["_WindowsJob"]:
            if not name:
                return None
            handle = cls._kernel32().OpenJobObjectW(
                _JOB_OBJECT_QUERY | _JOB_OBJECT_TERMINATE,
                False,
                name,
            )
            return cls(int(handle), name) if handle else None

        def assign(self, process_handle: Any) -> None:
            if not self._kernel32().AssignProcessToJobObject(
                self.handle, process_handle
            ):
                raise ctypes.WinError()

        def terminate(self, exit_code: int = 1) -> None:
            if not self._kernel32().TerminateJobObject(self.handle, int(exit_code)):
                error = ctypes.get_last_error()
                if error not in (0, 5, 6):
                    raise ctypes.WinError(error)

        def active_count(self) -> int:
            info = _BasicAccountingInformation()
            returned = ctypes.wintypes.DWORD()
            ok = self._kernel32().QueryInformationJobObject(
                self.handle,
                1,
                ctypes.byref(info),
                ctypes.sizeof(info),
                ctypes.byref(returned),
            )
            if not ok:
                return 0
            return int(info.ActiveProcesses)

        def close(self) -> None:
            if self.handle:
                self._kernel32().CloseHandle(self.handle)
                self.handle = 0

else:
    _WindowsJob = None  # type: ignore[assignment,misc]


def _send_graceful_signal(pid: int, *, record: Optional[_WorkerRecord] = None) -> str:
    """Try the platform's graceful signal without broad process matching."""
    if _IS_WINDOWS:
        ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
        if ctrl_break is not None:
            try:
                os.kill(int(pid), ctrl_break)
                return "ctrl_break"
            except (OSError, ProcessLookupError, SystemError):
                # Windows may raise SystemError(87) / WinError 87 when CTRL_BREAK_EVENT is unavailable.
                pass
        # ``Popen.terminate()`` is TerminateProcess on Windows: it is a hard
        # kill, not a graceful request. Leave it to the forced Job Object
        # phase so the exit envelope does not claim a forced stop was graceful.
        return "unavailable"

    pgid = None
    if record is not None and record.receipt.owner_kind == "process_group":
        try:
            pgid = int(record.receipt.owner_id)
        except (TypeError, ValueError):
            pgid = None
    if pgid:
        try:
            os.killpg(pgid, signal.SIGTERM)
            return "process_group_sigterm"
        except (OSError, ProcessLookupError):
            return "already_gone"
    try:
        os.kill(int(pid), signal.SIGTERM)
        return "pid_sigterm"
    except (OSError, ProcessLookupError):
        return "already_gone"


def _send_forced_signal(record: Optional[_WorkerRecord], pid: int, owner_kind: str, owner_id: str) -> str:
    if _IS_WINDOWS and owner_kind == "job_object":
        opened = record is None
        job = record.owner if record is not None else _WindowsJob.open(owner_id)
        if job is None:
            return "job_unavailable"
        try:
            job.terminate(1)
            return "job_terminate"
        except OSError:
            return "job_terminate_failed"
        finally:
            if opened:
                job.close()
    if not _IS_WINDOWS and owner_kind == "process_group":
        try:
            os.killpg(int(owner_id), getattr(signal, "SIGKILL", signal.SIGTERM))
            return "process_group_sigkill"
        except (OSError, ProcessLookupError):
            return "already_gone"
    return "no_owned_tree_handle"


def _tree_alive(
    record: Optional[_WorkerRecord],
    owner_kind: str,
    owner_id: str,
    pid: int,
    *,
    start_time: Optional[int] = None,
) -> bool:
    if _IS_WINDOWS and owner_kind == "job_object":
        opened = record is None
        job = record.owner if record is not None else _WindowsJob.open(owner_id)
        if job is None:
            # A dead wrapper does not prove that a persisted Job Object is
            # empty.  The one safe exception is a persisted exact root
            # identity that is now unambiguously absent: the monitor already
            # closed the final kill-on-close handle, so no member can remain
            # in that owned Job Object.  Live, recycled, and inaccessible
            # roots remain fail-closed below.
            if record is not None:
                start_time = record.receipt.start_time
            return not _root_identity_definitely_absent(pid, start_time)
        try:
            return job.active_count() > 0
        finally:
            if opened:
                job.close()
    if not _IS_WINDOWS and owner_kind == "process_group":
        return _group_alive(int(owner_id))
    return _pid_alive(pid)


def _close_record(pid: int, record: _WorkerRecord) -> None:
    with _RECORDS_LOCK:
        current = _RECORDS.get(int(pid))
        if current is not record:
            return
        _RECORDS.pop(int(pid), None)
    if record.owner is not None and _IS_WINDOWS:
        record.owner.close()


def _monitor_record(record: _WorkerRecord) -> None:
    pid = record.receipt.pid
    try:
        code = record.process.wait()
    except Exception:
        code = None
    with record.lock:
        record.root_exited = True
        record.exit_code = code
        record.known_descendants.update(_snapshot_descendants(pid))
    alive = _tree_alive(
        record,
        record.receipt.owner_kind,
        record.receipt.owner_id,
        pid,
        start_time=record.receipt.start_time,
    )
    if alive:
        cleanup_worker_tree(
            pid,
            start_time=record.receipt.start_time,
            owner_kind=record.receipt.owner_kind,
            owner_id=record.receipt.owner_id,
            envelope_path=record.receipt.envelope_path,
            reason="root-exit",
            _record=record,
        )
    else:
        # The worker wrapper writes the authoritative, schema-versioned exit
        # envelope immediately before it exits. Do not overwrite that
        # receipt after ``Popen.wait()`` returns: on Windows the monitor can
        # run after the wrapper has flushed its file, and replacing it with
        # the older lightweight cleanup record makes a typed outcome look
        # like ``worker_exit_unknown`` to the dispatcher.
        authoritative = False
        envelope_path = record.receipt.envelope_path
        if envelope_path:
            try:
                payload = json.loads(Path(envelope_path).read_text(encoding="utf-8"))
                authoritative = bool(
                    isinstance(payload, dict)
                    and payload.get("schema_version")
                    and payload.get("task_id")
                    and payload.get("failure_class")
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                authoritative = False
        if not authoritative:
            _update_envelope(
                record.receipt.envelope_path,
                schema=1,
                pid=pid,
                start_time=record.receipt.start_time,
                owner_kind=record.receipt.owner_kind,
                owner_id=record.receipt.owner_id,
                command=list(record.receipt.command),
                exit_code=code,
                exited_at=int(time.time()),
                graceful_cleanup=False,
                forced_cleanup=False,
                tree_cleanup=True,
                cleanup_reason="normal-exit",
                already_absent=False,
            )
        _close_record(pid, record)


class SpawnedWorker:
    """Small Popen-compatible handle returned by :func:`spawn_worker_process`."""

    def __init__(self, process: subprocess.Popen, receipt: WorkerReceipt):
        self._process = process
        self.receipt = receipt
        self.pid = int(process.pid)

    def poll(self):
        return self._process.poll()

    def wait(self, timeout: Optional[float] = None):
        return self._process.wait(timeout=timeout)

    def terminate(self):
        return self._process.terminate()

    def kill(self):
        return self._process.kill()


def spawn_worker_process(
    argv: list[str],
    *,
    envelope_path: Optional[str] = None,
    monitor: bool = True,
    popen_factory: Optional[Callable[..., subprocess.Popen]] = None,
    kill_on_parent_exit: bool = True,
    **popen_kwargs: Any,
) -> SpawnedWorker:
    """Spawn a worker with a durable process-group/job ownership handle.

    ``kill_on_parent_exit=False`` keeps the named Windows job alive after a
    one-shot dispatcher exits; the persisted job name remains available to
    later reconciliation while the worker writes its terminal envelope.
    """
    if popen_factory is None:
        # Resolve at call time so tests and embedding callers can replace
        # ``subprocess.Popen`` without being defeated by a definition-time
        # default captured during module import.
        popen_factory = subprocess.Popen
    owner = None
    if _IS_WINDOWS:
        owner = _WindowsJob.create(kill_on_close=kill_on_parent_exit)
        flags = int(popen_kwargs.get("creationflags", 0))
        popen_kwargs["creationflags"] = flags | _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW
        popen_kwargs.pop("start_new_session", None)
    else:
        popen_kwargs["start_new_session"] = True

    try:
        process = popen_factory(argv, **popen_kwargs)
        if _IS_WINDOWS:
            process_handle = getattr(process, "_handle", None)
            if process_handle is None:
                # Small Popen doubles used by callers often expose only
                # ``pid``. They cannot be assigned to a Job Object, but
                # accepting them keeps the wrapper's test seam compatible;
                # real CPython Popen instances always have ``_handle``.
                owner.close()
                owner = None
                owner_kind = None
                owner_id = None
            else:
                owner.assign(process_handle)  # noqa: SLF001 - CPython handle
                owner_kind = "job_object"
                owner_id = owner.name
        else:
            owner_kind = "process_group"
            owner_id = str(os.getpgid(int(process.pid)))
        start_time = (
            _wait_for_start_time(int(process.pid))
            if owner_kind is not None
            else None
        )
        if start_time is None and owner_kind is not None:
            raise RuntimeError(f"could not read creation time for worker pid {process.pid}")
        receipt = WorkerReceipt(
            pid=int(process.pid),
            start_time=int(start_time) if start_time is not None else None,
            owner_kind=owner_kind,
            owner_id=owner_id,
            envelope_path=str(envelope_path) if envelope_path else None,
            command=tuple(str(part) for part in argv),
        )
        record = _WorkerRecord(receipt=receipt, process=process, owner=owner)
        with _RECORDS_LOCK:
            _RECORDS[receipt.pid] = record
            _RECENT_RECEIPTS[receipt.pid] = (receipt, time.monotonic())
        if monitor:
            thread = threading.Thread(
                target=_monitor_record,
                args=(record,),
                name=f"hermes-kanban-worker-{receipt.pid}",
                daemon=True,
            )
            thread.start()
        return SpawnedWorker(process, receipt)
    except Exception:
        try:
            process = locals().get("process")
            if process is not None and process.poll() is None:
                if not _IS_WINDOWS:
                    try:
                        os.killpg(
                            os.getpgid(int(process.pid)),
                            getattr(signal, "SIGKILL", signal.SIGTERM),
                        )
                    except (OSError, ProcessLookupError):
                        pass
                else:
                    process.kill()
                process.wait(timeout=2)
        except Exception:
            pass
        if owner is not None:
            owner.close()
        raise


def get_worker_receipt(pid: int) -> dict[str, Any]:
    """Return persisted-spawn evidence for a live in-process worker."""
    with _RECORDS_LOCK:
        record = _RECORDS.get(int(pid))
        if record is not None:
            return record.receipt.as_dict()
        cached = _RECENT_RECEIPTS.get(int(pid))
        if cached is not None:
            receipt, observed_at = cached
            if time.monotonic() - observed_at <= _RECEIPT_RETENTION_SECONDS:
                return receipt.as_dict()
            _RECENT_RECEIPTS.pop(int(pid), None)
    return {}


def worker_identity_matches(pid: int, start_time: Optional[int]) -> tuple[bool, bool]:
    """Return ``(alive, matches)`` for a persisted worker identity.

    The caller can distinguish a dead original worker from a live PID that
    has already been recycled.  No signal is sent by this probe.
    """
    return _identity_state(int(pid), start_time)


def _record_for_pid(
    pid: int,
    *,
    start_time: Optional[int] = None,
    owner_kind: Optional[str] = None,
    owner_id: Optional[str] = None,
) -> Optional[_WorkerRecord]:
    with _RECORDS_LOCK:
        record = _RECORDS.get(int(pid))
        if record is None:
            return None
        receipt = record.receipt
        if start_time is not None and receipt.start_time != start_time:
            return None
        if owner_kind is not None and receipt.owner_kind != owner_kind:
            return None
        if owner_id is not None and receipt.owner_id != owner_id:
            return None
        return record


def cleanup_worker_tree(
    pid: int,
    *,
    start_time: Optional[int] = None,
    owner_kind: Optional[str] = None,
    owner_id: Optional[str] = None,
    envelope_path: Optional[str] = None,
    reason: str = "cleanup",
    grace_seconds: float = _DEFAULT_GRACE_SECONDS,
    force_wait_seconds: float = _DEFAULT_FORCE_WAIT_SECONDS,
    _record: Optional[_WorkerRecord] = None,
) -> dict[str, Any]:
    """Gracefully stop, then force-reap exactly one owned worker tree.

    The ownership handle is either the in-process registry record or the
    persisted ``(owner_kind, owner_id)`` pair.  A live PID with a mismatched
    creation time is treated as a recycled PID and is never signalled.
    """
    pid = int(pid) if pid else 0
    record = _record or _record_for_pid(
        pid,
        start_time=start_time,
        owner_kind=owner_kind,
        owner_id=owner_id,
    )
    if record is not None:
        receipt = record.receipt
        start_time = receipt.start_time if start_time is None else start_time
        owner_kind = receipt.owner_kind if owner_kind is None else owner_kind
        owner_id = receipt.owner_id if owner_id is None else owner_id
        envelope_path = receipt.envelope_path if envelope_path is None else envelope_path
    owner_kind = str(owner_kind or "")
    owner_id = str(owner_id or "")
    result: dict[str, Any] = {
        "pid": pid or None,
        "start_time": start_time,
        "owner_kind": owner_kind or None,
        "owner_id": owner_id or None,
        "cleanup_reason": reason,
        "identity_verified": False,
        "pid_reused": False,
        "graceful_cleanup": False,
        "forced_cleanup": False,
        "tree_cleanup": False,
        "already_absent": False,
        "survived_cleanup": False,
        "unsafe_to_reclaim": False,
        "graceful_action": None,
        "forced_action": None,
    }
    if pid <= 0:
        result["already_absent"] = True
        result["tree_cleanup"] = True
        return result

    # An owner handle without the root's creation fingerprint is not enough
    # to prove that a live group/job belongs to this worker.  This includes a
    # dead wrapper with surviving descendants: do not signal the group and do
    # not release the claim until a later check has verified absence.
    if start_time is None and owner_kind and owner_id:
        result["survived_cleanup"] = True
        result["unsafe_to_reclaim"] = True
        _update_envelope(
            envelope_path,
            pid=pid,
            start_time=None,
            owner_kind=owner_kind,
            owner_id=owner_id,
            cleanup_reason=reason,
            identity_verified=False,
            pid_reused=False,
            graceful_cleanup=False,
            forced_cleanup=False,
            tree_cleanup=False,
            survived_cleanup=True,
            unsafe_to_reclaim=True,
        )
        return result

    alive, matches = _identity_state(pid, start_time)
    if alive and start_time is not None and not matches:
        # The root PID now belongs to another process. Never signal it. The
        # ownership handle is still useful for proving whether descendants of
        # the original worker remain; a live owned tree keeps retry admission
        # closed until the next cleanup tick verifies its absence.
        if owner_kind and owner_id:
            tree_alive = _tree_alive(
                record,
                owner_kind,
                owner_id,
                pid,
                start_time=start_time,
            )
        else:
            # A mismatched PID without an ownership handle cannot prove that
            # descendants are gone, so fail closed rather than admitting a
            # duplicate beside an untracked old tree.
            tree_alive = True
        result["pid_reused"] = True
        result["tree_cleanup"] = not tree_alive
        result["already_absent"] = not tree_alive
        result["survived_cleanup"] = tree_alive
        result["unsafe_to_reclaim"] = tree_alive
        _update_envelope(
            envelope_path,
            pid=pid,
            start_time=start_time,
            cleanup_reason=reason,
            identity_verified=False,
            pid_reused=True,
            graceful_cleanup=False,
            forced_cleanup=False,
            tree_cleanup=result["tree_cleanup"],
            already_absent=result["already_absent"],
            survived_cleanup=result["survived_cleanup"],
            unsafe_to_reclaim=result["unsafe_to_reclaim"],
        )
        if not tree_alive and record is not None:
            _close_record(pid, record)
        return result
    # A persisted owner handle without the creation-time fingerprint is not
    # enough to prove that the live PID is the worker that owns the tree.  Do
    # not terminate a potentially recycled PID or its group/job.
    if alive and start_time is None:
        result["survived_cleanup"] = True
        result["unsafe_to_reclaim"] = True
        _update_envelope(
            envelope_path,
            pid=pid,
            owner_kind=owner_kind or None,
            owner_id=owner_id or None,
            cleanup_reason=reason,
            identity_verified=False,
            pid_reused=False,
            forced_cleanup=False,
            tree_cleanup=False,
            unsafe_to_reclaim=True,
        )
        return result
    if start_time is not None and (matches or not alive):
        result["identity_verified"] = True
    elif record is not None and record.receipt.start_time:
        result["identity_verified"] = bool(matches or not alive)
        if alive and not matches:
            result["pid_reused"] = True
            result["survived_cleanup"] = True
            return result

    tree_alive = _tree_alive(
        record,
        owner_kind,
        owner_id,
        pid,
        start_time=start_time,
    )
    if not tree_alive:
        result["already_absent"] = True
        result["tree_cleanup"] = True
        _update_envelope(
            envelope_path,
            pid=pid,
            start_time=start_time,
            owner_kind=owner_kind or None,
            owner_id=owner_id or None,
            exit_code=record.exit_code if record is not None else None,
            command=list(record.receipt.command) if record is not None else None,
            cleanup_reason=reason,
            identity_verified=result["identity_verified"],
            pid_reused=False,
            graceful_cleanup=False,
            forced_cleanup=False,
            tree_cleanup=True,
            already_absent=True,
        )
        if record is not None:
            _close_record(pid, record)
        return result

    if alive:
        result["graceful_action"] = _send_graceful_signal(pid, record=record)
        result["graceful_cleanup"] = result["graceful_action"] not in {
            None,
            "already_gone",
            "unavailable",
        }
    deadline = time.monotonic() + max(0.0, float(grace_seconds))
    while time.monotonic() < deadline and _tree_alive(
        record,
        owner_kind,
        owner_id,
        pid,
        start_time=start_time,
    ):
        time.sleep(_POLL_SECONDS)

    if _tree_alive(
        record,
        owner_kind,
        owner_id,
        pid,
        start_time=start_time,
    ):
        result["forced_action"] = _send_forced_signal(record, pid, owner_kind, owner_id)
        result["forced_cleanup"] = result["forced_action"] not in {
            None,
            "already_gone",
            "no_owned_tree_handle",
            "job_unavailable",
            "job_terminate_failed",
        }

    deadline = time.monotonic() + max(0.0, float(force_wait_seconds))
    while time.monotonic() < deadline and _tree_alive(
        record,
        owner_kind,
        owner_id,
        pid,
        start_time=start_time,
    ):
        time.sleep(_POLL_SECONDS)
    result["tree_cleanup"] = not _tree_alive(
        record,
        owner_kind,
        owner_id,
        pid,
        start_time=start_time,
    )
    result["survived_cleanup"] = not result["tree_cleanup"]
    _update_envelope(
        envelope_path,
        pid=pid,
        start_time=start_time,
        owner_kind=owner_kind or None,
        owner_id=owner_id or None,
        exit_code=record.exit_code if record is not None else None,
        command=list(record.receipt.command) if record is not None else None,
        cleanup_reason=reason,
        identity_verified=result["identity_verified"],
        pid_reused=result["pid_reused"],
        graceful_cleanup=result["graceful_cleanup"],
        forced_cleanup=result["forced_cleanup"],
        tree_cleanup=result["tree_cleanup"],
        survived_cleanup=result["survived_cleanup"],
        forced_action=result["forced_action"],
        cleanup_finished_at=int(time.time()),
    )
    if record is not None and result["tree_cleanup"]:
        _close_record(pid, record)
    return result


# Ensure a dispatcher shutdown does not leave named Job Objects alive. The job
# has KILL_ON_JOB_CLOSE, so this also safely reaps descendants on process exit.
def _close_all_records() -> None:  # pragma: no cover - interpreter shutdown
    with _RECORDS_LOCK:
        records = list(_RECORDS.values())
    for record in records:
        if record.owner is not None and _IS_WINDOWS:
            try:
                record.owner.close()
            except Exception:
                pass


import atexit

atexit.register(_close_all_records)

__all__ = [
    "SpawnedWorker",
    "WorkerReceipt",
    "cleanup_worker_tree",
    "get_worker_receipt",
    "worker_identity_matches",
    "spawn_worker_process",
]
