"""Exclusive Kanban database maintenance leases and writer admission.

Ordinary SQLite transactions remain serialized by SQLite.  This module adds a
separate advisory lock solely around database file lifecycle operations:
maintenance takes an exclusive lock while registered writers and write
transactions hold a shared lock.  The adjacent JSON files are diagnostics,
not the source of exclusivity; the OS lock is authoritative.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

_IS_WINDOWS = sys.platform == "win32"
_POLL_SECONDS = 0.02


class MaintenanceLeaseError(RuntimeError):
    """Base class for maintenance lease/admission failures."""


class MaintenanceLeaseBusyError(MaintenanceLeaseError):
    """Raised when maintenance cannot quiesce active holders in time."""

    def __init__(
        self,
        db_path: Path,
        *,
        holder: Optional[dict[str, Any]] = None,
        active_writers: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.db_path = db_path
        self.holder = holder or {}
        self.active_writers = active_writers or []
        detail = []
        if self.holder:
            detail.append(f"maintenance holder={self.holder}")
        if self.active_writers:
            detail.append(f"active writers={self.active_writers}")
        suffix = "; ".join(detail) or "another process holds the lease"
        super().__init__(
            f"Kanban database maintenance lease unavailable for {db_path}: {suffix}. "
            "Stop writers or wait for their transactions to finish, then retry."
        )


class MaintenanceInProgressError(MaintenanceLeaseError):
    """Raised when a new writer is admitted during exclusive maintenance."""


@dataclass(frozen=True)
class MaintenanceLease:
    db_path: Path
    action: str
    lease_id: str
    marker_path: Path


@dataclass
class WriterAdmission:
    """A held shared maintenance lock for one writer transaction."""

    record: dict[str, Any]
    handle: Any
    registration_path: Path
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.registration_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            _unlock(self.handle)
        except OSError:
            pass
        finally:
            try:
                self.handle.close()
            except OSError:
                pass


def _resolved(db_path: Path) -> Path:
    return Path(db_path).expanduser().resolve()


def _lock_path(db_path: Path) -> Path:
    path = _resolved(db_path)
    return path.with_name(path.name + ".maintenance.lock")


def _marker_path(db_path: Path) -> Path:
    path = _resolved(db_path)
    return path.with_name(path.name + ".maintenance.json")


def _writers_dir(db_path: Path) -> Path:
    path = _resolved(db_path)
    return path.with_name(path.name + ".writers")


def _metadata(*, action: str, lease_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "action": action,
        "lease_id": lease_id,
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "hostname": socket.gethostname(),
        "created_at": int(time.time()),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def current_maintenance_holder(db_path: Path) -> Optional[dict[str, Any]]:
    return _read_json(_marker_path(db_path))


def active_writer_registrations(db_path: Path) -> list[dict[str, Any]]:
    directory = _writers_dir(db_path)
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError:
        return []
    registrations: list[dict[str, Any]] = []
    for path in paths:
        record = _read_json(path)
        if record is not None:
            registrations.append(record)
    return registrations


def _try_lock(handle, *, exclusive: bool) -> bool:
    if _IS_WINDOWS:
        import msvcrt

        mode = msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), mode, 1)
            return True
        except OSError:
            return False

    import fcntl

    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _unlock(handle) -> None:
    if _IS_WINDOWS:
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _acquire(handle, *, exclusive: bool, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        if _try_lock(handle, exclusive=exclusive):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SECONDS)


@contextlib.contextmanager
def maintenance_lease(
    db_path: Path,
    *,
    action: str,
    timeout: float = 2.0,
    lease_id: Optional[str] = None,
) -> Iterator[MaintenanceLease]:
    """Acquire a bounded exclusive lease for DB file lifecycle operations."""
    path = _resolved(db_path)
    lock_path = _lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    marker = _marker_path(path)
    token = lease_id or f"km-{int(time.time()):x}-{secrets.token_hex(4)}"
    try:
        acquired = _acquire(handle, exclusive=True, timeout=timeout)
        if not acquired:
            raise MaintenanceLeaseBusyError(
                path,
                holder=current_maintenance_holder(path),
                active_writers=active_writer_registrations(path),
            )
        payload = _metadata(action=action, lease_id=token)
        payload["db_path"] = str(path)
        _atomic_write_json(marker, payload)
        yield MaintenanceLease(path, action, token, marker)
    finally:
        if acquired:
            try:
                current = _read_json(marker)
                if current is not None and current.get("lease_id") == token:
                    try:
                        marker.unlink(missing_ok=True)
                    except OSError:
                        pass
            finally:
                try:
                    _unlock(handle)
                except OSError:
                    pass
        try:
            handle.close()
        except OSError:
            pass


def acquire_writer_admission(
    db_path: Path,
    *,
    owner: str,
    timeout: float = 0.0,
) -> WriterAdmission:
    """Acquire and return one shared writer admission handle."""
    path = _resolved(db_path)
    lock_path = _lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if not _acquire(handle, exclusive=False, timeout=timeout):
            holder = current_maintenance_holder(path)
            raise MaintenanceInProgressError(
                f"Kanban database writes are paused for maintenance on {path}; "
                f"holder={holder or 'unknown'}"
            )
        token = f"kw-{os.getpid()}-{threading.get_ident()}-{secrets.token_hex(4)}"
        record = _metadata(action="writer", lease_id=token)
        record["owner"] = owner
        record["db_path"] = str(path)
        directory = _writers_dir(path)
        directory.mkdir(parents=True, exist_ok=True)
        registration_path = directory / f"{token}.json"
        _atomic_write_json(registration_path, record)
        return WriterAdmission(record, handle, registration_path)
    except Exception:
        try:
            _unlock(handle)
        except OSError:
            pass
        handle.close()
        raise


@contextlib.contextmanager
def writer_registration(
    db_path: Path,
    *,
    owner: str,
    timeout: float = 0.0,
) -> Iterator[dict[str, Any]]:
    """Register a writer capability and hold shared maintenance admission."""
    admission = acquire_writer_admission(db_path, owner=owner, timeout=timeout)
    try:
        yield admission.record
    finally:
        admission.close()
