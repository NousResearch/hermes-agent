"""Natural-language day/time schedule phrases -> 5-field cron (sibling of ``cron/jobs.py``).

``parse_schedule`` hands the text after an optional ``every `` prefix to
``natural_phrase_to_cron``; a ``None`` return means "not a day/time phrase", so the caller falls
back to its interval/cron/timestamp paths. Recognized shapes::

    monday 9am | day at 9am | weekdays at 9am | monday, wednesday and friday at 9am
    monday and friday at 9am and 5pm          (times sharing a minute -> one hour list)
    second monday at 9am | last friday of the month at 5pm | 1st and 3rd tuesday at 10:00

Ordinal weekdays compile to croniter's ``D#N`` (Nth weekday of the month) and ``LD`` (last
weekday of the month) day-of-week forms. Behaviour lives here, not in the facade: ``cron/jobs.py``
is past the file-lines ratchet.
"""

from __future__ import annotations

import re
from typing import Optional

# Cron weekday numbering is 0=Sunday … 6=Saturday (croniter's default).
_WEEKDAY_TO_CRON_DOW = {
    "sunday": "0", "sun": "0",
    "monday": "1", "mon": "1",
    "tuesday": "2", "tue": "2", "tues": "2",
    "wednesday": "3", "wed": "3", "weds": "3",
    "thursday": "4", "thu": "4", "thur": "4", "thurs": "4",
    "friday": "5", "fri": "5",
    "saturday": "6", "sat": "6",
}

# Keyword day-specs that expand to a cron weekday field.
_DAYSPEC_TO_CRON_DOW = {
    "day": "*", "daily": "*", "everyday": "*",
    "weekday": "1-5", "weekdays": "1-5",
    "weekend": "0,6", "weekends": "0,6",
}

# Ordinal of a weekday within its month; "last" maps to croniter's ``L<dow>``.
_ORDINALS = {
    "first": "1", "1st": "1", "second": "2", "2nd": "2", "third": "3", "3rd": "3",
    "fourth": "4", "4th": "4", "fifth": "5", "5th": "5", "last": "L",
}

# Filler words after an ordinal weekday: "second monday of the month", "... of every month".
_MONTH_FILLER = {"of", "the", "every", "each", "month", "monthly"}


def parse_clock_time(text: str) -> Optional[tuple]:
    """Parse ``9am``/``9:30am``/``14:00``/``7`` (bare 24h hour)/``noon``/``midnight`` into a
    24-hour ``(hour, minute)`` tuple, or None when unrecognized."""
    t = text.strip().lower().replace(" ", "")
    if not t:
        return None
    if t in ("noon", "midday"):
        return (12, 0)
    if t == "midnight":
        return (0, 0)
    match = re.match(r'^(\d{1,2})(?::(\d{2}))?(am|pm)?$', t)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    if hour > 23 or minute > 59:
        return None
    return (hour, minute)


def _ordinal_dow(tokens: list[str]) -> tuple[Optional[str], int]:
    """``first and third monday [of the month]`` -> (``1#1,1#3``, index past the phrase)."""
    ordinals: list[str] = []
    idx = 0
    while idx < len(tokens) and (tokens[idx] in _ORDINALS or tokens[idx] == "and"):
        if tokens[idx] != "and" and _ORDINALS[tokens[idx]] not in ordinals:
            ordinals.append(_ORDINALS[tokens[idx]])
        idx += 1
    if not ordinals or idx >= len(tokens) or tokens[idx] not in _WEEKDAY_TO_CRON_DOW:
        return None, 0
    day = _WEEKDAY_TO_CRON_DOW[tokens[idx]]
    idx += 1
    while idx < len(tokens) and tokens[idx] in _MONTH_FILLER:
        idx += 1
    return ",".join(f"L{day}" if o == "L" else f"{day}#{o}" for o in ordinals), idx


def _weekday_list_dow(tokens: list[str]) -> tuple[Optional[str], int]:
    """``monday and friday`` -> (``1,5``, index of the first non-day token)."""
    days: list[str] = []
    for i, tok in enumerate(tokens):
        if tok == "and":
            continue
        mapped = _WEEKDAY_TO_CRON_DOW.get(tok)
        if mapped is None:
            return (",".join(days) or None), i
        if mapped not in days:
            days.append(mapped)
    return (",".join(days) or None), len(tokens)


def _clock_fields(tokens: list[str]) -> Optional[tuple[str, str]]:
    """``9am and 5pm`` -> (``0``, ``9,17``); times must share a minute to fit one cron line."""
    times = []
    for chunk in " ".join(tokens).split(" and "):
        parsed = parse_clock_time(chunk)
        if parsed is None:
            return None
        times.append(parsed)
    if not times or len({minute for _, minute in times}) != 1:
        return None
    hours = sorted({hour for hour, _ in times})
    return str(times[0][1]), ",".join(str(h) for h in hours)


def natural_phrase_to_cron(rest: str) -> Optional[str]:
    """Convert ``<when> [at] <time>`` to a 5-field cron expr, or None when it is not one."""
    tokens = rest.lower().replace(",", " ").split()
    if not tokens:
        return None
    dow = _DAYSPEC_TO_CRON_DOW.get(tokens[0])
    idx = 1
    if dow is None:
        dow, idx = _ordinal_dow(tokens)
    if dow is None:
        dow, idx = _weekday_list_dow(tokens)
    if dow is None:
        return None
    time_tokens = tokens[idx:]
    if time_tokens and time_tokens[0] == "at":  # optional separator: "every day at 9am"
        time_tokens = time_tokens[1:]
    fields = _clock_fields(time_tokens) if time_tokens else None
    if fields is None:
        return None
    minute, hours = fields
    return f"{minute} {hours} * * {dow}"
