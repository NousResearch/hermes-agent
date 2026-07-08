"""Time helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone


def utc_now() -> datetime:
    """Current UTC timestamp."""

    return datetime.now(timezone.utc)


def today_utc() -> date:
    """Current UTC date."""

    return utc_now().date()
