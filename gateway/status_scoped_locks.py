"""Scoped-lock ownership and release, split from :mod:`gateway.status` (acquire stays there).

Helpers read status through the module at call time, preserving its patch seams.
The acquire path imports the ownership check lazily to avoid an import cycle.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Optional

from gateway import status as _status


def _scoped_lock_record_is_stale(existing: dict[str, Any], existing_pid: Optional[int]) -> bool:
    """Recover dead/reused/stopped owners. Gateway locks also validate gateway argv.

    An explicitly process-owned reservation can belong to a dashboard or setup
    worker. Unknown identity stays reserved instead of guessing that it is dead.
    """
    if existing_pid is None or not _status._pid_exists(existing_pid):
        return True
    recorded_start = existing.get("start_time")
    current_start = _status._get_process_start_time(existing_pid)
    if _status._start_times_conflict(recorded_start, current_start):
        return True
    metadata = existing.get("metadata")
    process_owner = isinstance(metadata, dict) and metadata.get("owner_kind") == "process"
    if not process_owner and not _status._looks_like_gateway_process(existing_pid):
        if _status._read_process_cmdline(existing_pid) is not None:
            return True
        if None in (recorded_start, current_start) and not _status._record_looks_like_gateway(existing):
            return True
    return _status._process_is_stopped(existing_pid)


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
