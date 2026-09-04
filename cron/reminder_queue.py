"""Inline reminder queue — transactional one-shot reminders.

A list of sticky notes with times + one alarm checker.  Not a cron registry
(that's for recurring cadences); this is a transactional queue for one-shot
"remind me Tue 8am to X" entries that fire once and expire themselves.

Storage: JSON file at ``<HERMES_HOME>/reminders/queue.json`` (pending entries)
+ append-only ``<HERMES_HOME>/reminders/fired.log`` (fired entries, one JSON
per line).  The fired log is the audit trail — entries are never deleted from
it, only appended.

Each pending entry:
    {
        "id":          str, uuid4 hex,
        "due_at":      str, ISO-8601 with offset (tz-aware),
        "message":     str, the reminder text rephrased relative to fire time,
        "origin":      {platform, chat_id, thread_id?},  # delivery target
        "status":      "pending",
        "created_at":  str, ISO-8601,
        "recurring":   null | {"kind": "weekly", "weekday": 1, "time": "08:00"} | {"kind": "daily", "time": "18:00"},  # poller re-arms via next_occurrence
    }

Operations: add / list / cancel / mark_fired / due_now.

Thread-safe (file lock via atomic rename on writes).  No LLM calls, no
network — pure local state.  The poller script (``reminder_poller.py``) reads
this queue and prints due reminders to stdout for the no_agent cron delivery
path.

Design: Argos issue #3.  This is generic Hermes functionality, not Argos
plugin work — reminders are not memory.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

_lock = threading.RLock()


def _queue_path() -> Path:
    """Path to the pending-reminders JSON file."""
    home = get_hermes_home()
    d = home / "reminders"
    d.mkdir(parents=True, exist_ok=True)
    return d / "queue.json"


def _fired_log_path() -> Path:
    """Path to the append-only fired-reminders log."""
    home = get_hermes_home()
    d = home / "reminders"
    d.mkdir(parents=True, exist_ok=True)
    return d / "fired.log"


def _load_queue() -> List[Dict[str, Any]]:
    """Load pending reminders from the queue file.  Missing/corrupt = empty."""
    path = _queue_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [e for e in data if isinstance(e, dict)]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_queue(entries: List[Dict[str, Any]]) -> None:
    """Atomically write the queue file (temp + rename)."""
    path = _queue_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _append_fired(entry: Dict[str, Any]) -> None:
    """Append a fired entry to the append-only log (one JSON per line)."""
    path = _fired_log_path()
    line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_reminder(
    due_at: datetime,
    message: str,
    origin: Optional[Dict[str, Any]] = None,
    recurring: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add a pending reminder to the queue.

    Args:
        due_at: Timezone-aware datetime for when to fire.
        message: The reminder text, rephrased relative to fire time
                 (e.g. "the plumber arrives today", not "tomorrow").
        origin: Delivery target — ``{platform, chat_id, thread_id?}``.
                None means "deliver to home channel" (fallback).
        recurring: If set, a structured recurrence rule — ``{"kind": "weekly",
                   "weekday": <0-6, Mon=0>, "time": "HH:MM"}`` or
                   ``{"kind": "daily", "time": "HH:MM"}``.  After firing, the
                   poller computes the next occurrence via ``next_occurrence``
                   and re-adds the entry with the same message/origin/rule.

    Returns:
        The created entry dict.
    """
    if due_at.tzinfo is None:
        raise ValueError("due_at must be timezone-aware")
    entry = {
        "id": uuid.uuid4().hex[:12],
        "due_at": due_at.isoformat(),
        "message": message,
        "origin": origin or {},
        "status": "pending",
        "created_at": _hermes_now().isoformat(),
        "recurring": recurring,
    }
    with _lock:
        entries = _load_queue()
        entries.append(entry)
        _save_queue(entries)
    return entry


def list_pending(
    *,
    sort_by_time: bool = True,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """List pending reminders, soonest first by default."""
    with _lock:
        entries = [e for e in _load_queue() if e.get("status") == "pending"]
    if sort_by_time:
        entries.sort(key=lambda e: e.get("due_at", ""))
    if limit:
        entries = entries[:limit]
    return entries


def cancel_reminder(reminder_id: str) -> bool:
    """Cancel a pending reminder by ID.  Returns True if found and removed."""
    with _lock:
        entries = _load_queue()
        before = len(entries)
        entries = [e for e in entries if not (
            e.get("id") == reminder_id and e.get("status") == "pending"
        )]
        if len(entries) == before:
            return False
        _save_queue(entries)
        return True


def cancel_by_query(query: str) -> List[Dict[str, Any]]:
    """Cancel pending reminders whose message contains *query* (case-insensitive).

    Returns the cancelled entries.  Used by the skill's "cancel the reminder
    about X" phrasing.
    """
    q = query.lower().strip()
    if not q:
        return []
    with _lock:
        entries = _load_queue()
        cancelled = []
        for e in entries:
            if e.get("status") != "pending":
                continue
            if q in (e.get("message", "")).lower():
                e["status"] = "cancelled"
                e["cancelled_at"] = _hermes_now().isoformat()
                cancelled.append(e)
        if cancelled:
            # Remove cancelled entries from the queue file.  Cancellations
            # are deliberately NOT appended to fired.log (cancel ≠ fire, and
            # that log is the delivery audit).  A separate cancels.log could
            # be added later if a cancellation audit is ever needed.
            remaining = [e for e in entries if e.get("status") == "pending"]
            _save_queue(remaining)
        return cancelled


def due_now(now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Return pending reminders whose due_at has passed.

    Does NOT mark them fired — the poller calls mark_fired after delivery.
    """
    if now is None:
        now = _hermes_now()
    with _lock:
        entries = _load_queue()
    due = []
    for e in entries:
        if e.get("status") != "pending":
            continue
        try:
            due_at = datetime.fromisoformat(e["due_at"])
        except (KeyError, ValueError):
            continue
        if due_at <= now:
            due.append(e)
    due.sort(key=lambda e: e.get("due_at", ""))
    return due


def mark_fired(reminder_id: str) -> bool:
    """Mark a reminder as fired and move it to the append-only fired log.

    Returns True if the entry was found and marked.  The entry is removed
    from the pending queue file and appended to fired.log with a
    ``fired_at`` timestamp.
    """
    with _lock:
        entries = _load_queue()
        target = None
        for e in entries:
            if e.get("id") == reminder_id and e.get("status") == "pending":
                target = e
                break
        if target is None:
            return False
        target["status"] = "fired"
        target["fired_at"] = _hermes_now().isoformat()
        _append_fired(target)
        remaining = [e for e in entries if e.get("id") != reminder_id]
        _save_queue(remaining)
        return True


def get_reminder(reminder_id: str) -> Optional[Dict[str, Any]]:
    """Get a single pending reminder by ID."""
    with _lock:
        for e in _load_queue():
            if e.get("id") == reminder_id:
                return e
    return None


def list_fired(limit: int = 100) -> List[Dict[str, Any]]:
    """List recently fired reminders from the append-only log (newest first)."""
    path = _fired_log_path()
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(lines[-limit:]):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------
# Time parsing helpers (used by the skill layer)
# ---------------------------------------------------------------------------

def parse_when(when_str: str, now: Optional[datetime] = None) -> datetime:
    """Parse a natural-language time expression into a timezone-aware datetime.

    Handles:
      - Absolute: "tuesday 8am", "tue 8am", "tomorrow 8am", "today 6pm",
        "2026-08-28 08:00", "aug 28 8am"
      - Relative: "in 20 minutes", "in 2 hours", "in 3 days"

    Uses the configured Hermes timezone (SAST/UTC+2 on this host).
    Returns a timezone-aware datetime.

    This is a deterministic parser — no LLM.  The skill layer's LLM does the
    initial extraction from the user's sentence; this function handles the
    structured "when" string the skill produces.
    """
    if now is None:
        now = _hermes_now()

    s = when_str.lower().strip()

    # --- relative: "in N <unit>" ---
    import re
    m = re.match(r"in\s+(\d+)\s+(second|seconds|sec|minute|minutes|min|hour|hours|hr|day|days|week|weeks)", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("second") or unit == "sec":
            delta = timedelta(seconds=n)
        elif unit.startswith("minute") or unit == "min":
            delta = timedelta(minutes=n)
        elif unit.startswith("hour") or unit == "hr":
            delta = timedelta(hours=n)
        elif unit.startswith("day"):
            delta = timedelta(days=n)
        elif unit.startswith("week"):
            delta = timedelta(weeks=n)
        else:
            raise ValueError(f"unknown unit: {unit}")
        return now + delta

    # --- absolute date + time ---
    # Day names → weekday numbers (0=Monday)
    day_map = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6,
    }

    # "tomorrow 8am", "today 6pm"
    m = re.match(r"(tomorrow|today)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s)
    if m:
        base_day = now.date()
        if m.group(1) == "tomorrow":
            base_day = base_day + timedelta(days=1)
        hour = int(m.group(2))
        minute = int(m.group(3) or 0)
        ampm = m.group(4)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        result = datetime.combine(base_day, datetime.min.time(), tzinfo=now.tzinfo)
        result = result.replace(hour=hour, minute=minute)
        return result

    # "tuesday 8am", "tue 08:00", "tuesday 8:30am"
    m = re.match(
        r"(monday|mon|tuesday|tue|tues|wednesday|wed|thursday|thu|thur|thurs|friday|fri|saturday|sat|sunday|sun)"
        r"\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        s,
    )
    if m:
        day_name = m.group(1)
        target_dow = day_map[day_name]
        hour = int(m.group(2))
        minute = int(m.group(3) or 0)
        ampm = m.group(4)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        # Find the next occurrence of target_dow
        days_ahead = (target_dow - now.weekday()) % 7
        if days_ahead == 0:
            # Today — if the time has passed, next week
            candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if candidate <= now:
                days_ahead = 7
        target_date = now.date() + timedelta(days=days_ahead)
        result = datetime.combine(target_date, datetime.min.time(), tzinfo=now.tzinfo)
        result = result.replace(hour=hour, minute=minute)
        return result

    # ISO-8601: "2026-08-28 08:00" or "2026-08-28T08:00:00"
    try:
        # Try parsing as ISO — if no tz, attach the configured tz
        result = datetime.fromisoformat(s)
        if result.tzinfo is None:
            result = result.replace(tzinfo=now.tzinfo)
        return result
    except ValueError:
        pass

    raise ValueError(f"could not parse time expression: {when_str!r}")


def next_occurrence(recurring: Dict[str, Any], after: datetime) -> datetime:
    """Next occurrence of a recurring rule, strictly after ``after``.

    Rules (structured, produced by the skill layer):
      ``{"kind": "weekly", "weekday": 0-6, "time": "HH:MM"}``  (Mon=0)
      ``{"kind": "daily", "time": "HH:MM"}``

    Returns a timezone-aware datetime in ``after``'s timezone.  The poller
    uses this to re-arm an entry after it fires (repeat-flag semantics — the
    queue is stateless about recurrence; the poller owns the loop).
    """
    if after.tzinfo is None:
        raise ValueError("after must be timezone-aware")
    if not isinstance(recurring, dict):
        raise ValueError(f"invalid recurring rule: {recurring!r}")

    try:
        hour, minute = (int(x) for x in str(recurring["time"]).split(":"))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid recurring time: {recurring!r}") from exc

    kind = recurring.get("kind")

    def candidate(day) -> datetime:
        return datetime.combine(
            day, datetime.min.time(), tzinfo=after.tzinfo
        ).replace(hour=hour, minute=minute)

    if kind == "daily":
        next_due = candidate(after.date())
        if next_due <= after:
            next_due = candidate(after.date() + timedelta(days=1))
        return next_due

    if kind == "weekly":
        try:
            target_dow = int(recurring["weekday"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid weekly weekday: {recurring!r}") from exc
        if not 0 <= target_dow <= 6:
            raise ValueError(f"weekly weekday out of range: {target_dow}")
        days_ahead = (target_dow - after.weekday()) % 7
        next_due = candidate(after.date() + timedelta(days=days_ahead))
        if next_due <= after:
            next_due = candidate(after.date() + timedelta(days=days_ahead + 7))
        return next_due

    raise ValueError(f"unknown recurring rule kind: {kind!r}")


def rephrase_for_fire_time(message: str, due_at: datetime, now: Optional[datetime] = None) -> str:
    """Rephrase a reminder message relative to fire time.

    "tomorrow" → "today", "next week" → "this week", etc.  This is a light
    heuristic — the skill layer's LLM does the heavy lifting; this is a
    fallback for the poller's verbatim delivery.
    """
    if now is None:
        now = _hermes_now()
    delta = due_at - now
    days = delta.days
    msg = message
    if days <= 0:
        msg = msg.replace("tomorrow", "today")
        msg = msg.replace("next week", "this week")
    elif days <= 1:
        msg = msg.replace("today", "tomorrow")
    return msg
