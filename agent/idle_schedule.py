"""Idle-job scheduling shared by the skill curator (``agent/curator.py``) and plugin idle tasks
(``hermes_cli/plugins_idle_tasks.py``): the interval gate, whose first observation seeds
``last_run_at`` and defers one interval, and the cross-process run claim."""

from __future__ import annotations

import contextlib
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CLAIM_STALE_SECONDS = 3600.0


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


def claim_run(lock: Path, stale_seconds: float = CLAIM_STALE_SECONDS) -> bool:
    """One run at a time across processes: the O_EXCL create of ``lock`` wins the claim; a claim
    older than ``stale_seconds`` is a crashed holder and is taken over. When the lock dir cannot be
    created, run anyway."""
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return True
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            if time.time() - lock.stat().st_mtime > stale_seconds:
                lock.unlink()
                return claim_run(lock, stale_seconds)
        except OSError:
            pass
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))
    return True


def release_run(lock: Path) -> None:
    with contextlib.suppress(OSError):
        lock.unlink()
