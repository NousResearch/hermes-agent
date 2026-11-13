"""Scoped-lock ownership and release, split from :mod:`gateway.status` (acquire stays there).

Helpers read status through the module at call time, preserving its patch seams.
The acquire path imports the ownership check lazily to avoid an import cycle.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Optional

from gateway import status as _status


def scoped_lock_pid_is_ours(record: dict[str, Any]) -> bool:
    """Whether the record's ``pid`` names a process in THIS PID namespace, so a local probe of
    it is evidence. A gateway in a container or under systemd ``PrivatePIDs=`` records its own
    namespace's PID (often 1): from outside, that number is init or nothing, and the live owner
    reads as dead (#123081). A stamp from another namespace is never probed; a record with no
    stamp (written before it existed, or on a platform without namespaces) keeps the pid probe."""
    from hermes_agent.hermes_state_pidns import persistent_record_pidns_checkable
    return persistent_record_pidns_checkable(record.get("pidns"))


def _same_owner(existing: dict[str, Any], holder: Any) -> bool:
    """Whether the home's ``gateway.lock`` record names the process that wrote *existing*: same pid in
    the same namespace, and the same start time when both carry one."""
    if not isinstance(holder, dict) or not holder.get("pid"):
        return True  # unpublished or torn (flock taken, record not yet written): unknown is not dead
    if (_status._pid_from_record(holder), holder.get("pidns")) != (_status._pid_from_record(existing),
                                                                    existing.get("pidns")):
        return False
    return not _status._start_times_conflict(existing.get("start_time"), holder.get("start_time"))


def _foreign_owner_is_live(existing: dict[str, Any]) -> bool:
    """A gateway in another PID namespace holds its home's ``gateway.lock`` flock for as long as it
    lives, and that flock answers from here where its PID cannot. A held flock proves only that SOME
    gateway runs on that home: the record it published must name this lock's owner, or a successor on
    the same home kept a dead predecessor's token reserved for its whole life. No home, or a
    process-kind owner (setup worker, dashboard) that never takes that flock, stays live: unknown is
    not dead."""
    from gateway.runtime_ownership import runtime_lock_held_elsewhere
    home, metadata = existing.get("hermes_home"), existing.get("metadata")
    if not isinstance(home, str) or not home.strip():
        return True
    if isinstance(metadata, dict) and metadata.get("owner_kind") == "process":
        return True
    if not runtime_lock_held_elsewhere(Path(home)):
        return False
    return _same_owner(existing, _status._read_gateway_lock_record(Path(home) / "gateway.lock"))


def _scoped_lock_record_is_stale(existing: dict[str, Any], existing_pid: Optional[int]) -> bool:
    """Recover dead/reused/stopped owners. Gateway locks also validate gateway argv.

    An explicitly process-owned reservation can belong to a dashboard or setup
    worker. Unknown identity stays reserved instead of guessing that it is dead.
    An owner in another PID namespace is judged by its home's runtime flock, never its PID.
    """
    if existing_pid is not None and not scoped_lock_pid_is_ours(existing):
        if not _foreign_owner_is_live(existing):
            return True  # its gateway.lock is free: that gateway is gone (SIGKILL, OOM, container kill)
        _status.logger.warning(
            "Scoped lock for %s is held by PID %s in another PID namespace (%s), whose gateway.lock "
            "in %s is held or cannot be checked; treating it as live.", existing.get("scope") or "?",
            existing_pid, existing.get("pidns"), existing.get("hermes_home") or "an unrecorded home")
        return False
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
    record = _status._read_json_file(lock_path) or {}
    if record.get("pid") == os.getpid() and scoped_lock_pid_is_ours(record):
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
            if _status._pid_from_record(record) != owner_pid or not scoped_lock_pid_is_ours(record) or (
                owner_start_time is not None and record.get("start_time") != owner_start_time
            ):
                continue
        with contextlib.suppress(OSError):
            lock_file.unlink(missing_ok=True)
            removed += 1
    return removed
