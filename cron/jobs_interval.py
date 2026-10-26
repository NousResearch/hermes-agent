"""Recurring interval schedules for ``parse_schedule`` (``cron/jobs.py``): the one-minute floor."""

from typing import Any, Optional


def _interval_schedule(minutes: int, original: Optional[str] = None) -> dict[str, Any]:
    """Build a recurring interval schedule, rejecting a sub-minute one.

    A zero (or sub-minute) interval is not a recurrence: the next run computes to
    ``last_run + 0 == last_run``, which is always in the past, so the job is perpetually
    due and re-fires on every scheduler tick. Both interval forms ("every 0m" and a bare
    "0m") land here, so the guard covers each of them. ``parse_schedule`` keeps its
    not-a-duration suppression off this call so the message reaches the user, and
    ``compute_next_run`` floors rows already on disk at one minute.
    """
    if minutes < 1:
        raise ValueError(
            f"Invalid interval '{original if original is not None else f'{minutes}m'}': "
            f"recurring interval must be at least 1 minute. "
            f"Use e.g. 'every 1m', 'every 30m', or 'every 2h'.")
    return {"kind": "interval", "minutes": minutes, "display": f"every {minutes}m"}
