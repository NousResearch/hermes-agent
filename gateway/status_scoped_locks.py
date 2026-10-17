"""Scoped-lock release, split from :mod:`gateway.status` (acquire stays there). One-way: this
module reads the status helpers through the module at call time (so tests that patch
``gateway.status`` keep steering them); :mod:`gateway.status` never imports it."""
from __future__ import annotations

import contextlib
import os
from typing import Optional

from gateway import status as _status


def release_scoped_lock(scope: str, identity: str) -> None:
    """Release a scope lock owned by this PID. No start_time equality check: on-disk null vs a live
    fingerprint would wedge reconnects."""
    lock_path = _status._get_scope_lock_path(scope, identity)
    if (_status._read_json_file(lock_path) or {}).get("pid") == os.getpid():
        _status._unlink_quietly(lock_path)


def release_all_scoped_locks(
    *, owner_pid: Optional[int] = None, owner_start_time: Optional[int] = None
) -> int:
    """Remove scoped lock files (--replace cleanup); returns the count removed. With ``owner_pid``
    only that gateway's records go (``owner_start_time`` narrows against PID reuse)."""
    lock_dir = _status._get_lock_dir()
    if not lock_dir.exists():
        return 0
    removed = 0
    for lock_file in lock_dir.glob("*.lock"):
        if owner_pid is not None:
            record = _status._read_json_file(lock_file) or {}
            if _status._pid_from_record(record) != owner_pid or (
                owner_start_time is not None and record.get("start_time") != owner_start_time
            ):
                continue
        with contextlib.suppress(OSError):
            lock_file.unlink(missing_ok=True)
            removed += 1
    return removed
