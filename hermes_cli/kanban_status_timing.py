"""Shared time formatting and refresh cadence for Kanban status surfaces."""

from __future__ import annotations

from collections.abc import Iterable


STATUS_SURFACE_REFRESH_SECONDS = 15
_MINUTE_SECONDS = 60


def _bucket_age(age: int) -> int:
    """Clamp an age and round it down to the visible refresh bucket."""
    return max(0, int(age)) // STATUS_SURFACE_REFRESH_SECONDS * STATUS_SURFACE_REFRESH_SECONDS


def _minute_second_text(age: int) -> str:
    minutes, seconds = divmod(max(0, int(age)), _MINUTE_SECONDS)
    if seconds:
        return f"{minutes}m {seconds}s"
    return f"{minutes}m"


def format_relative_age(age: int) -> str:
    """Format an age without hiding sub-minute movement in a zero-minute bucket."""
    bucketed = _bucket_age(age)
    if bucketed < STATUS_SURFACE_REFRESH_SECONDS:
        return "just now"
    if bucketed < _MINUTE_SECONDS:
        return f"{bucketed}s ago"
    return f"{_minute_second_text(bucketed)} ago"


def format_elapsed_age(age: int) -> str:
    """Format elapsed task time using the same visible buckets as relative age."""
    bucketed = _bucket_age(age)
    if bucketed < STATUS_SURFACE_REFRESH_SECONDS:
        return "under 15s"
    if bucketed < _MINUTE_SECONDS:
        return f"{bucketed}s"
    return _minute_second_text(bucketed)


def status_surface_refresh_period(metric_ages: Iterable[int | None]) -> int:
    """Return the single shared refresh cadence for non-terminal status cards.

    All displayed clocks are rounded to the same fifteen-second bucket, so one
    shared scheduler refresh lets the user observe movement without per-task
    timers. Missing clocks do not schedule a card that has nothing to render.
    """
    return STATUS_SURFACE_REFRESH_SECONDS if any(age is not None for age in metric_ages) else 0
