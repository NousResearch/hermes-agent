"""
Cron Gate Chain — configurable pre-flight checks for cron jobs.

Each gate is a pure function ``(job, now) → (passed: bool, reason: str)``.
Gates run sequentially; the first failure blocks the job.

Configured per-job via the ``gates`` dict in the job's config::

    gates:
      cooldown_minutes: 30       # minimum gap since last run
      max_daily: 12              # max runs in rolling 24h window
      active_hours: "09:00-18:00"  # only run within this time range
"""

import logging
from datetime import datetime, timedelta

from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)


def gate_cooldown(job: dict, now: datetime) -> tuple[bool, str]:
    """Skip if last run was within ``cooldown_minutes``.

    Returns ``(True, "")`` if the gate passes,
    ``(False, "reason")`` if it blocks.
    """
    minutes = (job.get("gates") or {}).get("cooldown_minutes")
    if not minutes:
        return True, ""
    last_run = job.get("last_run_at")
    if not last_run:
        return True, ""
    try:
        last_dt = datetime.fromisoformat(last_run)
        if hasattr(last_dt, "tzinfo") and last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=now.tzinfo)
    except Exception:
        return True, ""
    elapsed = (now - last_dt).total_seconds() / 60
    if elapsed < float(minutes):
        return False, f"cooldown: last run was {elapsed:.0f}m ago (minimum {minutes}m)"
    return True, ""


def gate_max_daily(job: dict, now: datetime) -> tuple[bool, str]:
    """Skip if run count in the last 24 hours exceeds ``max_daily``.

    Returns ``(True, "")`` if the gate passes,
    ``(False, "reason")`` if it blocks.
    """
    max_count = (job.get("gates") or {}).get("max_daily")
    if not max_count:
        return True, ""
    run_history = job.get("run_history")
    if not isinstance(run_history, list) or not run_history:
        return True, ""
    cutoff = now - timedelta(hours=24)
    recent = 0
    for entry in run_history:
        ts = entry.get("timestamp") if isinstance(entry, dict) else entry
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
            if hasattr(dt, "tzinfo") and dt.tzinfo is None:
                dt = dt.replace(tzinfo=now.tzinfo)
            if dt >= cutoff:
                recent += 1
        except Exception:
            continue
    if recent >= int(max_count):
        return False, f"max_daily: {recent} runs in last 24h (limit {max_count})"
    return True, ""


def gate_active_hours(job: dict, now: datetime) -> tuple[bool, str]:
    """Skip if current time is outside the configured window.

    Format: ``"HH:MM-HH:MM"`` (24h, e.g. ``"09:00-18:00"``).
    Supports ranges that cross midnight (e.g. ``"22:00-06:00"``).

    Returns ``(True, "")`` if the gate passes,
    ``(False, "reason")`` if it blocks.
    """
    window = (job.get("gates") or {}).get("active_hours")
    if not window or not isinstance(window, str) or "-" not in window:
        return True, ""
    try:
        start_str, end_str = window.split("-", 1)
        start_h, start_m = int(start_str.split(":")[0]), int(start_str.split(":")[1])
        end_h, end_m = int(end_str.split(":")[0]), int(end_str.split(":")[1])
    except (ValueError, IndexError):
        logger.warning("Job '%s': invalid active_hours format '%s'", job.get("id"), window)
        return True, ""
    current_minutes = now.hour * 60 + now.minute
    start_total = start_h * 60 + start_m
    end_total = end_h * 60 + end_m
    if start_total <= end_total:
        # Normal range (e.g. 09:00-18:00)
        if start_total <= current_minutes < end_total:
            return True, ""
    else:
        # Crosses midnight (e.g. 22:00-06:00)
        if current_minutes >= start_total or current_minutes < end_total:
            return True, ""
    return False, f"active_hours: current time {now.hour:02d}:{now.minute:02d} outside window {window}"


_GATES = [
    gate_cooldown,
    gate_max_daily,
    gate_active_hours,
]


def check_job_gates(job: dict) -> tuple[bool, str]:
    """Run the gate chain for a cron job.

    Iterates over all configured gates in order. The first gate that
    blocks short-circuits and returns the reason.

    Returns:
        ``(True, "")`` if all gates pass.
        ``(False, "reason")`` if a gate blocks.
    """
    gates_config = job.get("gates")
    if not gates_config or not isinstance(gates_config, dict):
        return True, ""  # no gates configured — always pass
    now = _hermes_now()
    for gate_fn in _GATES:
        passed, reason = gate_fn(job, now)
        if not passed:
            return False, reason
    return True, ""
