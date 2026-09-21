"""Bounded shutdown-replay for cron jobs interrupted by gateway shutdown.

When the gateway shuts down mid-run, in-flight jobs are marked ``interrupted``
(``mark_running_jobs_interrupted``) and their execution ledger rows terminalize
as ``failed`` with the interruption reason. The scheduler's at-most-once design
(``advance_next_run`` before ``run_job``) means that fire is simply lost — one
missed run beats a crash-loop burst. That is correct for runaway jobs, but a
*clean* shutdown interruption of an otherwise-healthy job loses a scheduled
delivery for no good reason.

This module replays such fires exactly once, subject to strict gates:

- Only jobs marked interrupted by a clean gateway shutdown (durable ledger
  ``failed`` row with the interruption error signature).
- Freshness window: only replays interruptions newer than the window (default
  6h) — replaying a stale fire would deliver stale digests.
- At most one replay per source execution (durable tombstone keyed by the
  source execution ID); a second recovery pass never replays again.
- Replay is suppressed when a completed execution already exists for the job's
  scheduled instant (idempotency: the run's side effects are known good).
- Never replays jobs whose latest execution failed for a NON-shutdown reason
  (provider errors, script failures — those belong to failure-streak handling).
- Job must still exist, be enabled, scheduled state, not in-flight, and not
  already claimed for a fire.

Replay re-fires through ``trigger_job`` (run-now intent), which the standard
claim/fire path treats like any manual run — so all existing fencing applies.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

# Replay only interruptions newer than this (seconds). 6h covers an overnight
# gateway restart without delivering stale content the next evening.
REPLAY_FRESHNESS_SECONDS = 6 * 3600.0

# Throttle for the sweep; called from the periodic dead-owner reap path.
REPLAY_SWEEP_INTERVAL_SECONDS = 300.0

# The durable signature written by the shutdown interruption path
# (_finish_interrupted_run / mark_running_jobs_interrupted via mark_job_run).
INTERRUPTION_SIGNATURES = (
    "Interrupted by gateway shutdown",
    "Interrupted by shutdown before terminal completion",
)

_TOMBSTONE_DB_LABEL = "cron/replay-tombstones.db"
_lock = threading.RLock()
_last_sweep_at: Optional[float] = None

import time  # noqa: E402  (used by throttle; kept near state for clarity)


def _tombstones_db():
    from hermes_constants import get_hermes_home

    return get_hermes_home().resolve() / "cron" / "replay-tombstones.db"


def _open_tombstones():
    import sqlite3

    # KENSEI NOTE: upstream retired cron/ledger.py (helpers folded into the shared
    # sqlite layer, see e24c8499); this inlines the same setup via open_db so the
    # fork's replay lane no longer imports a removed module.
    from cron.jobs import _ensure_cron_dir
    from hermes_cli.sqlite_util import open_db

    path = _tombstones_db()
    _ensure_cron_dir(path.parent)

    def _initialize_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS replay_tombstones (
                 source_execution_id TEXT PRIMARY KEY,
                 replayed_at TEXT NOT NULL
               )"""
        )

    return open_db(
        path, db_label=_TOMBSTONE_DB_LABEL, busy_timeout_ms=30000,
        synchronous_full=True, initialize=_initialize_schema,
    )


def _already_replayed(conn, source_execution_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM replay_tombstones WHERE source_execution_id=?",
        (source_execution_id,),
    ).fetchone()
    return row is not None


def _record_replayed(conn, source_execution_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO replay_tombstones (source_execution_id, replayed_at) VALUES (?, ?)",
        (source_execution_id, _hermes_now().isoformat()),
    )


def _parse_iso(value: Any):
    from datetime import datetime

    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _interruption_is_fresh(finished_at: Any) -> bool:
    ts = _parse_iso(finished_at)
    if ts is None:
        return False
    try:
        from datetime import timezone

        age = _hermes_now().timestamp() - ts.timestamp()
        return 0 <= age <= REPLAY_FRESHNESS_SECONDS
    except Exception:
        return False


def _latest_execution_is_shutdown_interruption(job_id: str) -> Optional[Dict[str, Any]]:
    """Return the interrupted execution row iff the job's LATEST ledger row is
    a fresh shutdown interruption (and nothing newer succeeded/failed)."""
    from cron.executions import latest_execution

    row = latest_execution(job_id)
    if not row:
        return None
    if row.get("status") not in ("failed", "unknown"):
        return None
    error = str(row.get("error") or "")
    if not any(sig in error for sig in INTERRUPTION_SIGNATURES):
        return None
    if not _interruption_is_fresh(row.get("finished_at") or row.get("claimed_at")):
        return None
    return row


def _completed_execution_exists_for_instant(job_id: str, scheduled_instant: Any) -> bool:
    """Idempotency gate: a completed execution for the same scheduled instant
    means the fire's side effects are already known good — never replay."""
    if not scheduled_instant:
        return False
    try:
        from cron.executions import list_executions

        for row in list_executions(job_id=job_id, limit=50):
            if (
                row.get("status") == "completed"
                and row.get("scheduled_instant")
                and str(row["scheduled_instant"]) == str(scheduled_instant)
            ):
                return True
        return False
    except Exception:
        return True  # fail safe: inability to check → do not replay


def _job_replayable(job: Dict[str, Any], job_id: str) -> bool:
    from cron.jobs import is_terminal_job

    if not job or is_terminal_job(job):
        return False
    if not job.get("enabled"):
        return False
    # Do not double-fire a job that is currently in flight or claimed.
    from cron.scheduler import _running_job_ids

    if job_id in _running_job_ids:
        return False
    return True


def replay_sweep(max_replays: int = 3) -> List[str]:
    """Find fresh shutdown-interrupted jobs and re-fire each at most once.

    Returns the list of job IDs replayed this sweep.
    """
    global _last_sweep_at
    now_mono = time.monotonic()
    with _lock:
        if (
            _last_sweep_at is not None
            and now_mono - _last_sweep_at < REPLAY_SWEEP_INTERVAL_SECONDS
        ):
            return []
        _last_sweep_at = now_mono

    replayed: List[str] = []
    try:
        from cron.executions import list_executions
        from cron.jobs import load_jobs, resolve_job_ref
    except Exception as exc:  # pragma: no cover - import failure is fatal to sweep
        logger.debug("Replay sweep unavailable: %s", exc)
        return []

    try:
        rows = list_executions(limit=100)
    except Exception as exc:
        logger.debug("Replay sweep could not read executions: %s", exc)
        return []

    candidates: Dict[str, Dict[str, Any]] = {}
    seen_jobs: set = set()
    for row in rows:
        job_id = str(row.get("job_id") or "")
        if not job_id or job_id in seen_jobs:
            continue
        seen_jobs.add(job_id)
        error = str(row.get("error") or "")
        if row.get("status") in ("failed", "unknown") and any(
            sig in error for sig in INTERRUPTION_SIGNATURES
        ):
            candidates[job_id] = row

    if not candidates:
        return []

    try:
        jobs = load_jobs()
    except Exception as exc:
        logger.debug("Replay sweep could not load jobs: %s", exc)
        return []
    by_id = {}
    for job in jobs:
        jid = str(job.get("id") or "")
        if jid:
            by_id[jid] = job

    conn = _open_tombstones()
    try:
        for job_id, row in candidates.items():
            if len(replayed) >= max_replays:
                break
            if not _interruption_is_fresh(row.get("finished_at") or row.get("claimed_at")):
                continue
            if _already_replayed(conn, str(row["id"])):
                continue
            # Latest row must STILL be the interruption (nothing newer happened).
            latest = _latest_execution_is_shutdown_interruption(job_id)
            if not latest or str(latest.get("id")) != str(row["id"]):
                continue
            # Idempotency: same scheduled instant already completed → skip.
            if _completed_execution_exists_for_instant(
                job_id, row.get("scheduled_instant")
            ):
                _record_replayed(conn, str(row["id"]))
                continue
            job = by_id.get(job_id)
            if not isinstance(job, dict) or not _job_replayable(job, job_id):
                continue
            try:
                from cron.jobs import trigger_job

                updated = trigger_job(job_id)
                if updated is not None:
                    _record_replayed(conn, str(row["id"]))
                    replayed.append(job_id)
                    logger.warning(
                        "Replayed cron job '%s' after shutdown interruption of "
                        "execution %s (bounded replay, tombstoned)",
                        job.get("name") or job_id, row["id"])
            except Exception as exc:
                logger.warning("Replay of job %s failed: %s", job_id, exc)
        conn.commit()
    finally:
        conn.close()
    return replayed
