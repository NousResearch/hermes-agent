"""Basic Chinese time range parsing for attendance queries."""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from .models import TimeRange


def parse_time_range(text: str | None, *, today: date | None = None) -> TimeRange:
    base = today or date.today()
    query = (text or "").strip()

    explicit = _parse_explicit_range(query)
    if explicit:
        return explicit

    explicit_day = _parse_single_day(query)
    if explicit_day:
        return explicit_day

    if "昨天" in query:
        day = base - timedelta(days=1)
        return TimeRange(day, day + timedelta(days=1), "昨天")
    if "今天" in query:
        return TimeRange(base, base + timedelta(days=1), "今天")
    if any(term in query for term in ("上周", "上星期")):
        this_monday = base - timedelta(days=base.weekday())
        start = this_monday - timedelta(days=7)
        return TimeRange(start, this_monday, "上周")
    if any(term in query for term in ("本周", "这周", "本星期")):
        start = base - timedelta(days=base.weekday())
        return TimeRange(start, start + timedelta(days=7), "本周")
    if "上个月" in query or "上月" in query:
        first_this_month = base.replace(day=1)
        last_prev_month = first_this_month - timedelta(days=1)
        start = last_prev_month.replace(day=1)
        return TimeRange(start, first_this_month, "上个月")
    if "本月" in query or "这个月" in query:
        start = base.replace(day=1)
        return TimeRange(start, _add_month(start), "本月")

    month_match = re.search(r"(?:(\d{4})年)?(\d{1,2})\s*月份?", query)
    if month_match:
        year = int(month_match.group(1) or base.year)
        month = int(month_match.group(2))
        start = date(year, month, 1)
        return TimeRange(start, _add_month(start), f"{year}-{month:02d}")

    start = base - timedelta(days=6)
    return TimeRange(start, base + timedelta(days=1), "最近一周")


def _parse_explicit_range(text: str) -> TimeRange | None:
    match = re.search(
        r"(\d{4}-\d{1,2}-\d{1,2})\s*(?:到|至|~|--|—)\s*(\d{4}-\d{1,2}-\d{1,2})",
        text,
    )
    if not match:
        return None
    start = _parse_date(match.group(1))
    end = _parse_date(match.group(2)) + timedelta(days=1)
    return TimeRange(start, end, f"{start.isoformat()} 到 {(end - timedelta(days=1)).isoformat()}")


def _parse_single_day(text: str) -> TimeRange | None:
    match = re.search(r"\d{4}-\d{1,2}-\d{1,2}", text)
    if not match:
        return None
    day = _parse_date(match.group(0))
    return TimeRange(day, day + timedelta(days=1), day.isoformat())


def _parse_date(value: str) -> date:
    year, month, day = [int(part) for part in value.split("-")]
    return date(year, month, day)


def _add_month(value: date) -> date:
    year = value.year + (1 if value.month == 12 else 0)
    month = 1 if value.month == 12 else value.month + 1
    # Keep the helper safe if later callers pass non-first-of-month dates.
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)
