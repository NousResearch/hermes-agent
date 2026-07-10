"""Durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue.
An attempt left claimed or running when a scheduler process restarts is marked
``unknown`` because the new process cannot prove whether its side effects ran.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now
from utils import atomic_replace

EXECUTIONS_FILE = get_hermes_home().resolve() / "cron" / "executions.json"
EXECUTIONS_LOCK_FILE = get_hermes_home().resolve() / "cron" / ".executions.lock"
_lock = threading.RLock()
_lock_state = threading.local()
_PROCESS_ID = uuid.uuid4().hex


@contextlib.contextmanager
def _ledger_lock():
    depth = getattr(_lock_state, "depth", 0)
    if depth:
        _lock_state.depth = depth + 1
        try:
            yield
        finally:
            _lock_state.depth -= 1
        return

    with _lock:
        EXECUTIONS_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(EXECUTIONS_LOCK_FILE, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            elif msvcrt:  # pragma: no cover - Windows
                lock_file.seek(0)
                if not lock_file.read(1):
                    lock_file.write("0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            _lock_state.depth = 1
            yield
        finally:
            _lock_state.depth = 0
            if fcntl:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            elif msvcrt:  # pragma: no cover - Windows
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            lock_file.close()


def _load_unlocked() -> List[Dict[str, Any]]:
    if not EXECUTIONS_FILE.exists():
        return []
    try:
        data = json.loads(EXECUTIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records = data.get("executions", []) if isinstance(data, dict) else []
    return records if isinstance(records, list) else []


def _save_unlocked(records: List[Dict[str, Any]]) -> None:
    EXECUTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(EXECUTIONS_FILE.parent), prefix=".executions_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "executions": records}, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(tmp_name, EXECUTIONS_FILE)
        try:
            os.chmod(EXECUTIONS_FILE, 0o600)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def create_execution(job_id: str, *, source: str) -> Dict[str, Any]:
    """Persist a claimed attempt before it is submitted for execution."""
    now = _hermes_now().isoformat()
    record: Dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "job_id": str(job_id),
        "source": str(source),
        "process_id": _PROCESS_ID,
        "pid": os.getpid(),
        "status": "claimed",
        "claimed_at": now,
        "started_at": None,
        "finished_at": None,
        "error": None,
    }
    with _ledger_lock():
        records = _load_unlocked()
        records.append(record)
        _save_unlocked(records)
    return copy.deepcopy(record)


def _transition(execution_id: str, status: str, **updates: Any) -> Optional[Dict[str, Any]]:
    with _ledger_lock():
        records = _load_unlocked()
        for record in records:
            if record.get("id") != execution_id:
                continue
            record["status"] = status
            record.update(updates)
            _save_unlocked(records)
            return copy.deepcopy(record)
    return None


def mark_execution_running(execution_id: str) -> Optional[Dict[str, Any]]:
    return _transition(
        execution_id,
        "running",
        started_at=_hermes_now().isoformat(),
    )


def finish_execution(
    execution_id: str,
    *,
    success: bool,
    error: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    return _transition(
        execution_id,
        "completed" if success else "failed",
        finished_at=_hermes_now().isoformat(),
        error=None if success else (str(error) if error else "unknown failure"),
    )


def recover_interrupted_executions() -> int:
    """Classify prior in-flight attempts as unknown; never enqueue a retry."""
    now = _hermes_now().isoformat()
    changed = 0

    def _owner_is_live(record: Dict[str, Any]) -> bool:
        try:
            owner_pid = int(record.get("pid"))
        except (TypeError, ValueError):
            return False
        if owner_pid <= 0:
            return False
        if owner_pid == os.getpid():
            return True
        try:
            os.kill(owner_pid, 0)  # windows-footgun: ok -- liveness probe
            return True
        except OSError:
            return False

    with _ledger_lock():
        records = _load_unlocked()
        for record in records:
            if record.get("status") not in {"claimed", "running"}:
                continue
            # Multiple scheduler surfaces can start in one process. Their
            # startup recovery must not relabel work this same process is
            # currently executing. A real restart imports this module in a new
            # process and therefore has a distinct process id.
            if record.get("process_id") == _PROCESS_ID or _owner_is_live(record):
                continue
            record["status"] = "unknown"
            record["finished_at"] = now
            record["error"] = (
                "Scheduler restarted before this execution reached a durable "
                "terminal state; whether side effects ran is unknown."
            )
            changed += 1
        if changed:
            _save_unlocked(records)
    return changed


def list_executions(*, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
    with _ledger_lock():
        records = copy.deepcopy(_load_unlocked())
    if job_id is not None:
        records = [record for record in records if record.get("job_id") == job_id]
    records.sort(key=lambda record: str(record.get("claimed_at", "")), reverse=True)
    return records


def latest_execution(job_id: str) -> Optional[Dict[str, Any]]:
    records = list_executions(job_id=job_id)
    return records[0] if records else None
