"""Minimal iCalendar (.ics) parsing for Hermes Hub — stdlib only.

Understands enough of RFC 5545 to render personal calendars read-only:
VEVENT blocks, all-day and timed DTSTART (with or without TZID — times are
taken at face value, dates are what matter for a dashboard), and basic
recurrence: RRULE FREQ=DAILY/WEEKLY/MONTHLY/YEARLY with INTERVAL, COUNT,
UNTIL and (for WEEKLY) BYDAY. EXDATE is honored. Anything fancier renders
as its first occurrence.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

WEEKDAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}
MAX_OCCURRENCES = 400  # per event, guards against pathological rules


def _unfold(text: str) -> list[str]:
    """RFC 5545 line unfolding: continuation lines start with space/tab."""
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _parse_prop(line: str) -> tuple[str, dict, str] | None:
    if ":" not in line:
        return None
    head, value = line.split(":", 1)
    parts = head.split(";")
    name = parts[0].upper()
    params = {}
    for param in parts[1:]:
        if "=" in param:
            key, val = param.split("=", 1)
            params[key.upper()] = val
    return name, params, value


def _parse_dt(value: str) -> tuple[date, str | None] | None:
    """Returns (date, HH:MM or None). Timezone suffixes are ignored."""
    value = value.strip()
    try:
        if "T" in value:
            stamp = value.rstrip("Z")
            parsed = datetime.strptime(stamp[:15], "%Y%m%dT%H%M%S")
            return parsed.date(), parsed.strftime("%H:%M")
        return datetime.strptime(value[:8], "%Y%m%d").date(), None
    except ValueError:
        return None


def _parse_rrule(value: str) -> dict:
    rule = {}
    for part in value.split(";"):
        if "=" in part:
            key, val = part.split("=", 1)
            rule[key.upper()] = val
    return rule


def _expand(start: date, time_str: str | None, rule: dict,
            window_start: date, window_end: date, exdates: set[date]) -> list[date]:
    freq = rule.get("FREQ", "").upper()
    if freq not in ("DAILY", "WEEKLY", "MONTHLY", "YEARLY"):
        return [start] if window_start <= start <= window_end else []

    interval = max(1, int(rule.get("INTERVAL", "1") or 1))
    count = int(rule["COUNT"]) if rule.get("COUNT", "").isdigit() else None
    until = None
    if rule.get("UNTIL"):
        parsed = _parse_dt(rule["UNTIL"])
        until = parsed[0] if parsed else None

    if freq == "WEEKLY" and rule.get("BYDAY"):
        weekdays = {WEEKDAYS[d] for d in rule["BYDAY"].split(",") if d in WEEKDAYS}
    else:
        weekdays = None

    # Without a COUNT, history before the window is irrelevant — fast-forward
    # (COUNT semantics require counting occurrences from DTSTART, so keep those).
    if count is None and start < window_start:
        behind = (window_start - start).days
        if freq == "DAILY":
            start += timedelta(days=(behind // interval) * interval)
        elif freq == "WEEKLY":
            weeks_behind = behind // (7 * interval)
            start += timedelta(weeks=weeks_behind * interval)

    occurrences: list[date] = []
    current = start
    emitted = 0  # occurrences since DTSTART (drives COUNT)
    for _ in range(20_000):  # hard iteration guard (~55 years of daily steps)
        if len(occurrences) >= MAX_OCCURRENCES or (count and emitted >= count):
            break
        if until and current > until:
            break
        if current > window_end:
            break
        emit = True
        if weekdays is not None and current.weekday() not in weekdays:
            emit = False
        if emit:
            emitted += 1
            if current >= window_start and current not in exdates:
                occurrences.append(current)
        # advance
        if freq == "DAILY":
            current += timedelta(days=interval)
        elif freq == "WEEKLY":
            if weekdays is not None:
                nxt = current + timedelta(days=1)
                # skip ahead by (interval-1) weeks when wrapping past Sunday
                if nxt.weekday() == 0 and interval > 1:
                    nxt += timedelta(weeks=interval - 1)
                current = nxt
            else:
                current += timedelta(weeks=interval)
        elif freq == "MONTHLY":
            month = current.month - 1 + interval
            year = current.year + month // 12
            month = month % 12 + 1
            try:
                current = current.replace(year=year, month=month)
            except ValueError:  # e.g. Jan 31 → Feb
                current = date(year, month, 28)
        else:  # YEARLY
            try:
                current = current.replace(year=current.year + interval)
            except ValueError:  # Feb 29
                current = current.replace(year=current.year + interval, day=28)
    return occurrences


def parse_ics(text: str, calendar_name: str, window_start: date, window_end: date) -> list[dict]:
    """Extract events within [window_start, window_end], expanded and sorted."""
    events: list[dict] = []
    in_event = False
    props: list[tuple[str, dict, str]] = []

    for line in _unfold(text):
        stripped = line.strip()
        if stripped.upper() == "BEGIN:VEVENT":
            in_event = True
            props = []
        elif stripped.upper() == "END:VEVENT":
            in_event = False
            events.extend(_event_from_props(props, calendar_name, window_start, window_end))
        elif in_event:
            parsed = _parse_prop(line)
            if parsed:
                props.append(parsed)

    events.sort(key=lambda e: (e["date"], e.get("time") or ""))
    return events


def _event_from_props(props, calendar_name, window_start, window_end) -> list[dict]:
    summary = ""
    start = None
    time_str = None
    rule: dict = {}
    exdates: set[date] = set()
    for name, params, value in props:
        if name == "SUMMARY":
            summary = value.replace("\\,", ",").replace("\\;", ";").replace("\\n", " ").strip()
        elif name == "DTSTART":
            parsed = _parse_dt(value)
            if parsed:
                start, time_str = parsed
        elif name == "RRULE":
            rule = _parse_rrule(value)
        elif name == "EXDATE":
            for chunk in value.split(","):
                parsed = _parse_dt(chunk)
                if parsed:
                    exdates.add(parsed[0])
    if not summary or start is None:
        return []
    return [
        {
            "date": day.isoformat(),
            "time": time_str,
            "title": summary[:120],
            "calendar": calendar_name,
        }
        for day in _expand(start, time_str, rule, window_start, window_end, exdates)
    ]
