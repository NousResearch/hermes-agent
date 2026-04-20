"""Copilot job reaper — monitors running processes and enforces idle TTL.

Call ``reap(db)`` periodically (e.g. every 30 s from a timer or cron job)
to:
  1. Detect dead processes on *running* jobs and transition them to *idle*.
  2. Close *idle* jobs whose TTL has expired.
  3. Skip jobs owned by a human (``owner == 'human'``).
"""

import logging
import os
import time
from typing import Dict, List

from hermes_state import SessionDB

logger = logging.getLogger(__name__)


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process is still running (POSIX ``kill -0``)."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — treat as alive.
        return True


def reap_dead_processes(db: SessionDB) -> List[Dict]:
    """Find running jobs whose PID has exited and mark them idle.

    Returns the list of jobs that were transitioned.
    """
    reaped = []
    jobs = db.list_copilot_jobs(state="running", limit=100)
    for job in jobs:
        if job["owner"] == "human":
            continue
        pid = job.get("pid")
        if pid and not _is_pid_alive(pid):
            try:
                db.mark_copilot_job_idle(job["id"])
                db.record_copilot_job_event(
                    job["id"],
                    event_type="reaper_process_dead",
                    payload_json=f'{{"pid": {pid}}}',
                )
                reaped.append(job)
                logger.info(
                    "Reaper: job %s pid %d is dead → idle", job["id"], pid
                )
            except ValueError:
                pass  # already transitioned
    return reaped


def reap_expired_idle(db: SessionDB) -> List[Dict]:
    """Close idle jobs whose TTL has expired.

    Returns the list of jobs that were closed.
    """
    closed = []
    now = time.time()
    jobs = db.list_copilot_jobs(state="idle", limit=100)
    for job in jobs:
        if job["owner"] == "human":
            continue
        idle_since = job.get("idle_since") or 0
        ttl = job.get("idle_ttl_seconds") or 300
        if idle_since and (now - idle_since) >= ttl:
            try:
                db.close_copilot_job(job["id"], reason="ttl_expired")
                db.record_copilot_job_event(
                    job["id"],
                    event_type="reaper_ttl_expired",
                    payload_json=f'{{"idle_seconds": {int(now - idle_since)}, "ttl": {ttl}}}',
                )
                closed.append(job)
                logger.info(
                    "Reaper: job %s idle %ds > TTL %ds → closed",
                    job["id"], int(now - idle_since), ttl,
                )
            except ValueError:
                pass  # already transitioned
    return closed


def reap_stale_pending(db: SessionDB, max_age: int = 3600) -> List[Dict]:
    """Close pending jobs older than *max_age* seconds that never started.

    Prevents orphan pending records from accumulating.
    """
    closed = []
    now = time.time()
    jobs = db.list_copilot_jobs(state="pending", limit=100)
    for job in jobs:
        created = job.get("created_at") or 0
        if created and (now - created) >= max_age:
            try:
                db.close_copilot_job(job["id"], reason="stale_pending")
                db.record_copilot_job_event(
                    job["id"],
                    event_type="reaper_stale_pending",
                    payload_json=f'{{"age_seconds": {int(now - created)}}}',
                )
                closed.append(job)
                logger.info(
                    "Reaper: pending job %s age %ds → closed",
                    job["id"], int(now - created),
                )
            except ValueError:
                pass
    return closed


def reap(db: SessionDB, *, pending_max_age: int = 3600) -> Dict[str, List[Dict]]:
    """Run all reaper checks. Returns a summary dict.

    Keys:
      ``dead_processes``  — running jobs whose PID died → idle
      ``ttl_expired``     — idle jobs past TTL → closed
      ``stale_pending``   — old pending jobs → closed
    """
    return {
        "dead_processes": reap_dead_processes(db),
        "ttl_expired": reap_expired_idle(db),
        "stale_pending": reap_stale_pending(db, max_age=pending_max_age),
    }
