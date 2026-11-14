"""Idle-job scheduling shared by the skill curator (``agent/curator.py``) and plugin idle tasks
(``hermes_cli/plugins_idle_tasks.py``): the interval gate, whose first observation seeds
``last_run_at`` and defers one interval, and the cross-process run claim."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CLAIM_STALE_SECONDS = 3600.0
# This process image: ``os.execvp`` (hermes_cli/relaunch.py) keeps the pid and its start time, so a
# claim the image before the exec took would otherwise look held by a live owner until this exits.
_IMAGE = uuid.uuid4().hex


def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts) if ts else None
    except (TypeError, ValueError):
        return None


def interval_due(
    state: dict[str, Any], interval_hours: float, *, now: datetime, seed_summary: str,
    save: Callable[[dict[str, Any]], None],
) -> bool:
    """True once ``interval_hours`` have passed since ``state["last_run_at"]``. Without a
    ``last_run_at`` the first observation seeds it to ``now`` (``last_run_summary`` =
    ``seed_summary``, persisted through ``save``) and defers one interval, so a fresh install or
    update never runs a job on its first tick."""
    last = parse_iso(state.get("last_run_at"))
    if last is None:
        state["last_run_at"] = now.isoformat()
        state["last_run_summary"] = seed_summary
        try:
            save(state)
        except Exception:  # pragma: no cover — best-effort persistence
            logger.debug("Failed to seed last_run_at", exc_info=True)
        return False
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return (now - last) >= timedelta(hours=interval_hours)


def _machine() -> str:
    """This machine's current boot, which a pid is only meaningful within: the container's or VM's
    instantiation (boot id + PID 1 start) where ``/proc`` exists, otherwise (macOS, Windows) the host
    name and boot time from psutil. ``""`` when neither is available."""
    from gateway.drain_control import current_instantiation_epoch

    epoch = current_instantiation_epoch()
    if epoch:
        return epoch
    with contextlib.suppress(Exception):
        import socket

        import psutil  # type: ignore

        return f"host:{socket.gethostname()}:boot:{int(psutil.boot_time())}"
    return ""


def _owner_record() -> dict[str, Any]:
    from gateway.status import get_process_start_time

    pid = os.getpid()
    return {"pid": pid, "start": get_process_start_time(pid), "machine": _machine() or None, "image": _IMAGE}


def _read_owner(lock: Path) -> Optional[dict[str, Any]]:
    """The claim's owner record; ``None`` when unreadable or a bare pid (a claim of an older Hermes)."""
    try:
        data = json.loads(lock.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _owner_alive(owner: Optional[dict[str, Any]]) -> Optional[bool]:
    """Whether the claim's owner still runs; ``None`` when that cannot be proven either way: no
    record, another machine, or no start time to tell the owner from a process that reused its pid."""
    pid, start, machine = (owner or {}).get("pid"), (owner or {}).get("start"), (owner or {}).get("machine")
    if type(pid) is not int or pid <= 0 or type(start) is not int or not machine or machine != _machine():
        return None
    if pid == os.getpid() and (owner or {}).get("image") not in (None, _IMAGE):
        return False  # taken by this pid's image before an exec, which never released it
    from gateway.status import _pid_exists, get_process_start_time

    live_start = get_process_start_time(pid)
    if live_start is not None and live_start != start:
        return False  # the pid now belongs to another process
    if not _pid_exists(pid):  # gone, or a zombie
        return False
    return True if live_start is not None else None


def _abandoned(lock: Path, stale_seconds: float) -> bool:
    """A claim is taken over when its owner is provably dead; a live owner's claim never is. When
    the owner cannot be proven either way, a claim older than ``stale_seconds`` counts as crashed."""
    alive = _owner_alive(_read_owner(lock))
    if alive is not None:
        return not alive
    return time.time() - lock.stat().st_mtime > stale_seconds


def _create(lock: Path) -> bool:
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_owner_record()))
    return True


def _unlink_guard(lock: Path):
    """Every unlink of ``lock`` (a takeover or a release) happens under this cross-process file
    lock, so the claim a takeover judged abandoned is the one it removes: two claimants never both
    win, and a release never removes a claim another process has since taken."""
    from tools.skill_usage import skill_file_lock

    return skill_file_lock(lock.with_name(lock.name + ".guard"))


def claim_run(lock: Path, stale_seconds: float = CLAIM_STALE_SECONDS) -> bool:
    """One run at a time across processes: the O_EXCL create of ``lock`` wins the claim. The claim
    records its owner (pid, process start, machine); a claim whose owner is provably dead is taken
    over at once, one whose owner cannot be checked (another machine, an older Hermes's bare pid)
    once it is older than ``stale_seconds``. When the lock dir cannot be created, run anyway."""
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return True
    if _create(lock):
        return True
    try:
        with _unlink_guard(lock):
            if _create(lock):  # released meanwhile
                return True
            if not _abandoned(lock, stale_seconds):
                return False
            lock.unlink()
            return _create(lock)
    except OSError:
        return False


def release_run(lock: Path) -> None:
    """Drop this process's claim; a claim another process took over (as abandoned) stays."""
    with contextlib.suppress(OSError), _unlink_guard(lock):
        owner = _read_owner(lock)
        if owner is None or (owner.get("pid") == os.getpid() and owner.get("image") in (None, _IMAGE)):
            lock.unlink()
