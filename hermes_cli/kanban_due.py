"""Kanban time input: epoch coercion, the duration grammar, and task due dates.

Every surface that sets ``tasks.due_at`` (``hermes kanban create/edit --due``, the dashboard
API, the ``kanban_create`` tool) goes through :func:`parse_due`, so they all accept the same
forms. ``due_at`` is INTEGER epoch seconds, an absolute UTC instant, like every other
``tasks`` timestamp. Wall-clock input (a naive ISO timestamp, a bare date) is read in
the profile's configured timezone (``hermes_time``), the same rule cron applies to a naive
one-shot timestamp, except that an explicit naive time a DST change skips or repeats is refused
instead of guessed. Leaf module: it imports nothing from ``kanban_db``.
"""

from __future__ import annotations

import math
import re
import time
from datetime import date, datetime, timedelta
from datetime import time as dtime
from typing import Any, Callable, Optional

import hermes_time

# Year 9999. Also rejects millisecond epochs (a JS ``Date.now()``), which would land in year 50k+.
_MAX_EPOCH = 253402300799
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CLEAR_WORDS = frozenset({"none", "null", "-", "clear"})
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
DUE_FORMS_HELP = (
    "ISO-8601 timestamp (2026-11-20T17:00, 2026-11-20T17:00:00Z; no offset = configured "
    "timezone), a date (2026-11-20 = end of that day), 'in <duration>' (in 2h, in 3d), "
    "epoch seconds, or 'none' to clear"
)


def to_epoch(val: Any) -> Optional[int]:
    """Epoch seconds from int/float/numeric string/ISO-8601; None for empty/invalid."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    # ISO-8601 fallback (e.g. '2026-05-10T15:00:00Z')
    try:
        return int(datetime.fromisoformat(s).timestamp())
    except (ValueError, OSError):
        return None


def parse_duration(val: Any) -> Optional[int]:
    """``30s`` / ``5m`` / ``2h`` / ``1d`` or a raw integer → seconds; None for empty input;
    ValueError on malformed input."""
    if val is None or val == "":
        return None
    s = str(val).strip().lower()
    try:
        return int(s)  # bare integer → seconds
    except ValueError:
        pass
    if not (s and s[-1] in _DURATION_UNITS):
        raise ValueError(f"malformed duration {val!r} (expected 30s, 5m, 2h, 1d, or a number)")
    try:
        n = float(s[:-1])
    except ValueError as exc:
        raise ValueError(f"malformed duration {val!r}") from exc
    seconds = n * _DURATION_UNITS[s[-1]]
    # "infh" / "nanm" parse as floats, and "1e308d" overflows to inf only after scaling.
    if not math.isfinite(seconds):
        raise ValueError(f"malformed duration {val!r} (must be a finite number)")
    return int(seconds)


def _localize(naive: datetime, fold: int = 0) -> datetime:
    """Attach the configured zone to a wall-clock value; server-local (with its DST rules,
    not today's fixed offset) when no ``timezone`` is configured. Lenient: a skipped or
    repeated wall time resolves by ``fold`` (for computed bounds, not user input)."""
    tz = hermes_time.get_timezone()
    naive = naive.replace(fold=fold)
    return naive.replace(tzinfo=tz) if tz is not None else naive.astimezone()


def _wall_clock(epoch: float) -> datetime:
    tz = hermes_time.get_timezone()
    local = datetime.fromtimestamp(epoch, tz) if tz is not None else datetime.fromtimestamp(epoch)
    return local.replace(tzinfo=None)


def _localize_strict(naive: datetime, raw: str) -> datetime:
    """:func:`_localize` for a user-typed naive time: a time a DST change skips does not
    exist and one it repeats is ambiguous, so both are refused rather than silently shifted."""
    zone = hermes_time.get_timezone_name() or "server-local time"
    early = _localize(naive, fold=0)
    if _wall_clock(early.timestamp()) != naive:
        raise ValueError(f"invalid due date {raw!r}: that local time does not exist in {zone} "
                         "(skipped by a DST change); pick another time or add a UTC offset")
    if early.utcoffset() != _localize(naive, fold=1).utcoffset():
        raise ValueError(f"invalid due date {raw!r}: that local time occurs twice in {zone} "
                         "(DST change); add an explicit UTC offset, e.g. -04:00 or -05:00")
    return early


def _local_date(now: Optional[float]) -> date:
    if now is None:
        return hermes_time.now().date()
    tz = hermes_time.get_timezone()
    return datetime.fromtimestamp(now, tz).date() if tz is not None else datetime.fromtimestamp(now).date()


def _checked_epoch(value: float) -> int:
    if not 0 <= value <= _MAX_EPOCH:
        raise ValueError(f"due date {value!r} is out of range (expected epoch SECONDS)")
    return int(value)


def _parse_due_text(s: str, now: Optional[float]) -> Optional[int]:
    if s.lower() in _CLEAR_WORDS:
        return None
    if s.isdigit():
        return _checked_epoch(int(s))
    if s.lower().startswith("in "):
        seconds = parse_duration(s[3:].strip())
        if seconds is None or seconds < 0:
            raise ValueError(f"invalid due date {s!r}: 'in' needs a positive duration")
        if seconds > _MAX_EPOCH:  # before the add: a huge int would overflow the float sum
            raise ValueError(f"invalid due date {s!r}: duration is out of range")
        return _checked_epoch((time.time() if now is None else now) + seconds)
    if _DATE_ONLY.match(s):
        # A bare date is due BY that day: its last second, so it is not overdue during it.
        day = date.fromisoformat(s)
        return _checked_epoch(_localize(datetime.combine(day, dtime(23, 59, 59))).timestamp())
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise ValueError(f"invalid due date {s!r}: expected {DUE_FORMS_HELP}") from None
    return _checked_epoch((dt if dt.tzinfo is not None else _localize_strict(dt, s)).timestamp())


def parse_due(value: Any, *, now: Optional[float] = None) -> Optional[int]:
    """Normalise a due-date input to epoch seconds; None = no due date. ValueError on junk."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("due date must be a timestamp, not a boolean")
    if isinstance(value, float):  # never truncate 1.5 to epoch 1
        raise ValueError(f"due date {value!r} must be whole epoch seconds or a date string")
    if isinstance(value, int):
        return _checked_epoch(value)
    s = str(value).strip()
    return _parse_due_text(s, now) if s else None


def due_iso(epoch: Optional[int]) -> Optional[str]:
    """ISO-8601 rendering of ``due_at`` in the configured timezone (offset included)."""
    if epoch is None:
        return None
    tz = hermes_time.get_timezone()
    dt = datetime.fromtimestamp(epoch, tz) if tz is not None else datetime.fromtimestamp(epoch).astimezone()
    return dt.isoformat(timespec="seconds")


def today_window(now: Optional[float] = None) -> tuple[int, int]:
    """``[start, end)`` epoch bounds of the current day in the configured timezone."""
    day = _local_date(now)
    start = _localize(datetime.combine(day, dtime.min))
    end = _localize(datetime.combine(day + timedelta(days=1), dtime.min))
    return int(start.timestamp()), int(end.timestamp())


def _overdue(now: Optional[float]) -> tuple[str, list]:
    # A finished task is never overdue, whatever its date.
    cutoff = int(time.time() if now is None else now)
    return "due_at IS NOT NULL AND due_at < ? AND status NOT IN ('done', 'archived')", [cutoff]


def _due_today(now: Optional[float]) -> tuple[str, list]:
    return "due_at >= ? AND due_at < ?", list(today_window(now))


def due_fields(due_at: Optional[int], status: str, now: Optional[float] = None) -> dict[str, Optional[str]]:
    """Display fields for a due date in the configured timezone, with the SAME overdue/today
    rules as :data:`DUE_FILTERS`: ``due_at_iso``, ``due_at_local`` (``YYYY-MM-DDTHH:MM``, an
    HTML ``datetime-local`` value) and ``due_state`` (overdue / today / later / None)."""
    if due_at is None:
        return {"due_at_iso": None, "due_at_local": None, "due_state": None}
    now = time.time() if now is None else now
    start, end = today_window(now)
    if due_at < now and status not in ("done", "archived"):
        state = "overdue"
    else:
        state = "today" if start <= due_at < end else "later"
    local = _wall_clock(due_at).isoformat(timespec="minutes")
    return {"due_at_iso": due_iso(due_at), "due_at_local": local, "due_state": state}


# ``hermes kanban list --due <filter>``: name -> (now) -> (SQL predicate, params).
DUE_FILTERS: dict[str, Callable[[Optional[float]], tuple[str, list]]] = {
    "overdue": _overdue,
    "today": _due_today,
    "any": lambda now: ("due_at IS NOT NULL", []),
    "none": lambda now: ("due_at IS NULL", []),
}

# Merged into ``kanban_db.VALID_SORT_ORDERS``; undated tasks always sort last.
DUE_SORT_ORDERS: dict[str, str] = {
    "due": "due_at ASC NULLS LAST, priority DESC, created_at ASC",
    "due-desc": "due_at DESC NULLS LAST, priority DESC, created_at ASC",
}


def due_filter_sql(name: str, now: Optional[float] = None) -> tuple[str, list]:
    """SQL predicate + params for a :data:`DUE_FILTERS` name; ValueError for an unknown one."""
    build = DUE_FILTERS.get(str(name).strip().lower())
    if build is None:
        raise ValueError(f"due filter must be one of {sorted(DUE_FILTERS)}")
    return build(now)
