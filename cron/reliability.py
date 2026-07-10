"""Cron reliability safeguards: detection helpers for the failure modes that
let scheduled jobs silently disappear, stop firing, or fail without a trace.

These are deliberately small, pure functions over plain job dicts (plus a
persisted job *census* maintained by ``cron.jobs.save_jobs``) so they can be
called cheaply from the ticker loop, ``hermes cron status``, or a dashboard,
and unit-tested without touching the scheduler.

Four dimensions, mirroring the incident that motivated them:

* **job-loss detection** — jobs that vanished from ``jobs.json`` relative to the
  last sanctioned save (``detect_dropped_jobs``).
* **missed-run detection** — enabled jobs whose ``next_run_at`` is long past, so
  the scheduler evidently never fired them (``find_missed_runs``).
* **visible failure state** — jobs carrying a failure signal, surfaced even when
  ``enabled=False`` (which ``list_jobs`` hides by default) (``find_failed_jobs``).
* **durable delivery status** — recorded by ``cron.jobs.mark_job_run``; surfaced
  here via the ``last_delivery_error`` / ``last_delivery_status`` failure signals.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("cron.reliability")

# A job whose next_run_at is older than this many seconds — after allowing for
# the ticker interval and per-job schedule grace — is treated as a missed run.
# 10 minutes is well above the 60s tick cadence and any legitimate between-tick
# lag, so it fires only when the scheduler genuinely failed to run the job.
DEFAULT_MISSED_RUN_GRACE_SECONDS = 600


def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse a stored ISO timestamp into an aware datetime, or None.

    Delegates timezone normalization to ``cron.jobs._ensure_aware`` — the single
    tz-normalization point the rest of cron uses — so both naive and aware
    values are normalized to the configured Hermes zone and comparisons never
    mix offsets.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    from cron.jobs import _ensure_aware

    return _ensure_aware(dt)


def _is_schedulable(job: Dict[str, Any]) -> bool:
    """True when a job is expected to keep firing (enabled and not paused)."""
    if not job.get("enabled", True):
        return False
    if job.get("state") == "paused" or job.get("paused_at"):
        return False
    return True


def find_missed_runs(
    jobs: Iterable[Dict[str, Any]],
    now: datetime,
    grace_seconds: float = DEFAULT_MISSED_RUN_GRACE_SECONDS,
) -> List[Dict[str, Any]]:
    """Return recurring jobs whose ``next_run_at`` is overdue by > grace.

    An overdue ``next_run_at`` on a *recurring* (cron/interval) job means the
    ticker should have fired it and advanced the timestamp but did not — the
    process was down, the tick wedged, or the job was skipped.

    Excluded (their past ``next_run_at`` is normal, not a miss):
      * paused/disabled jobs — nothing to be late for.
      * jobs missing/with an unparseable ``next_run_at``.
      * jobs currently in flight (a ``run_claim``/``fire_claim`` is stamped for
        the duration of a run, and one-shot ``next_run_at`` is intentionally
        left in the past while the agent runs), which would otherwise false-alarm
        every tick for a healthy long-running job.
      * one-shot (``kind == "once"``) jobs — single-fire; ``advance_next_run``
        never moves their ``next_run_at`` forward, so a past value is expected.
    """
    missed: List[Dict[str, Any]] = []
    for job in jobs:
        if not _is_schedulable(job):
            continue
        if job.get("run_claim") or job.get("fire_claim"):
            continue
        if (job.get("schedule") or {}).get("kind") == "once":
            continue
        nxt = _parse_dt(job.get("next_run_at"))
        if nxt is None:
            continue
        if (now - nxt).total_seconds() > grace_seconds:
            missed.append(job)
    return missed


def find_failed_jobs(jobs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return jobs carrying a failure signal, regardless of ``enabled``.

    ``list_jobs(include_disabled=False)`` hides ``enabled=False`` jobs, so a job
    that failed and was auto-disabled/completed would silently drop off the
    default listing. This surfaces every job whose last run errored, whose
    delivery failed, or whose state is ``error`` — the visible-failure signal.
    """
    failed: List[Dict[str, Any]] = []
    for job in jobs:
        if (
            job.get("state") == "error"
            or job.get("last_status") == "error"
            or job.get("last_error")
            or job.get("last_delivery_error")
        ):
            failed.append(job)
    return failed


def detect_dropped_jobs(
    expected_ids: Iterable[str],
    current_jobs: Iterable[Dict[str, Any]],
) -> List[str]:
    """Return ids in ``expected_ids`` that are absent from ``current_jobs``.

    ``expected_ids`` is normally the persisted census (``cron.jobs.known_job_ids``)
    — the id set of the last sanctioned save. Any id missing from the live store
    disappeared out-of-band (external edit/truncation, corruption repair that
    dropped records, or a partial write), which is exactly the silent job loss
    this guards against. Returned sorted and de-duplicated so the result (and the
    log line built from it) is deterministic even though ``expected_ids`` is
    typically an unordered set.
    """
    present = {j.get("id") for j in current_jobs if j.get("id")}
    return sorted({jid for jid in expected_ids if jid and jid not in present})


def audit_cron_health(
    jobs: List[Dict[str, Any]],
    now: datetime,
    *,
    expected_ids: Optional[Iterable[str]] = None,
    grace_seconds: float = DEFAULT_MISSED_RUN_GRACE_SECONDS,
) -> Dict[str, List[str]]:
    """Summarize the three detectable failure modes as id lists (for logging)."""
    return {
        "missed_runs": [j["id"] for j in find_missed_runs(jobs, now, grace_seconds) if j.get("id")],
        "failed": [j["id"] for j in find_failed_jobs(jobs) if j.get("id")],
        "dropped": detect_dropped_jobs(expected_ids or [], jobs),
    }


_DIMENSION_MESSAGES = (
    ("dropped", "cron job-loss detected: %d job(s) missing from jobs.json since last save: %s"),
    ("missed_runs", "cron missed runs: %d job(s) overdue (next_run_at long past): %s"),
    ("failed", "cron jobs in failure state: %d job(s): %s"),
)

# Per-store memory of the last-warned id-set for each dimension, so a persistent
# condition (a job stuck in failure state, a lingering completed one-shot) is
# logged when it appears/changes, not re-flooded every tick. Keyed by cron dir.
_last_warned: Dict[str, Dict[str, frozenset]] = {}


def log_cron_health() -> Dict[str, List[str]]:
    """Best-effort health audit of the active cron store; warn on new findings.

    Reads the live jobs plus the persisted census and emits a WARNING per
    dimension only when its set of flagged ids *changes* since the last audit, so
    a standing condition doesn't re-flood the log every tick. Never raises — the
    ticker loop relies on this being fully inert on any failure (including a
    corrupt jobs.json, which is exactly the input it exists to observe).
    """
    empty = {"missed_runs": [], "failed": [], "dropped": []}
    try:
        from cron.jobs import known_job_ids, load_jobs, _hermes_now, _current_cron_store

        jobs = load_jobs()
        report = audit_cron_health(jobs, _hermes_now(), expected_ids=known_job_ids())
        key = str(_current_cron_store().cron_dir)
        prev = _last_warned.get(key, {})
        current: Dict[str, frozenset] = {}
        for dim, message in _DIMENSION_MESSAGES:
            ids = report[dim]
            current[dim] = frozenset(ids)
            if ids and current[dim] != prev.get(dim):
                logger.warning(message, len(ids), ", ".join(str(i) for i in ids))
        _last_warned[key] = current
        return report
    except Exception:
        logger.debug("cron health audit failed", exc_info=True)
        return empty
