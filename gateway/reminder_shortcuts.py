"""Deterministic gateway shortcuts for simple reminder requests.

These helpers intentionally bypass the LLM for high-confidence reminder
creation on chat platforms. Reminder creation is a transactional control-plane
operation: the confirmation must describe exactly the one job that was just
created, not summarize prior conversation history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from typing import Optional


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_NUMBER_PATTERN = (
    r"\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
)
_UNIT_PATTERN = r"m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|wk|wks|week|weeks"

# "Remind me laundry in 1 minute"
_RELATIVE_SUBJECT_BEFORE_RE = re.compile(
    rf"""
    ^\s*
    (?:please\s+)?
    (?:remind\s+me|reminder)
    \s*
    (?:again\s*)?
    (?P<subject>.+?)
    \s+in\s+
    (?P<num>{_NUMBER_PATTERN})
    \s*
    (?P<unit>{_UNIT_PATTERN})
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "Remind me in 2 minutes, test 03" / "Remind me in 2 minutes for test 03"
_RELATIVE_SUBJECT_AFTER_RE = re.compile(
    rf"""
    ^\s*
    (?:please\s+)?
    (?:remind\s+me|reminder)
    \s*
    (?:again\s*)?
    in\s+
    (?P<num>{_NUMBER_PATTERN})
    \s*
    (?P<unit>{_UNIT_PATTERN})
    (?:\s*(?:,|:|;|-|\bfor\b|\babout\b|\bto\b)\s*(?P<subject>.+?))?
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# STT often omits punctuation: "Remind me in 2 minutes test 03".
_RELATIVE_SUBJECT_AFTER_BARE_RE = re.compile(
    rf"""
    ^\s*
    (?:please\s+)?
    (?:remind\s+me|reminder)
    \s*
    (?:again\s*)?
    in\s+
    (?P<num>{_NUMBER_PATTERN})
    \s*
    (?P<unit>{_UNIT_PATTERN})
    \s+(?P<subject>.+?)
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CLOCK_RE = r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>a\.?m\.?|p\.?m\.?)"

# "Remind me tomorrow at 5am acrylic" / "Reminder tomorrow at 5:05am graphics"
_TOMORROW_AT_RE = re.compile(
    rf"""
    ^\s*
    (?:please\s+)?
    (?:remind\s+me|reminder)
    \s+
    tomorrow
    \s+(?:at\s+)?
    {_CLOCK_RE}
    \s+(?P<subject>.+?)
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_NEXT_WEEK_RE = re.compile(
    rf"""
    ^\s*
    (?:please\s+)?
    (?:remind\s+me|reminder)
    \s*
    (?:again\s*)?
    next\s+week
    (?:\s*(?:,|:|;|-|\bfor\b|\babout\b|\bto\b)\s*(?P<subject>.+?))?
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_VOICE_TRANSCRIPT_RE = re.compile(
    r"Here's\s+what\s+they\s+said:\s*[\"“](?P<text>.*?)[\"”]\s*\]?$",
    re.IGNORECASE | re.DOTALL,
)

# --- Reply-to-reminder snooze/reschedule/close vocabulary ----------------------
#
# When a user replies to a fired reminder, the reply is a control-plane action
# on *that* reminder, not a new-reminder request. The reply forms below are
# recognized WITHOUT requiring a "remind me" prefix and WITHOUT an inline
# subject — the subject is always inherited from the quoted reminder.

_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
_WEEKDAY_PATTERN = (
    r"monday|mon|tuesday|tues|tue|wednesday|wed|thursday|thurs|thur|thu|"
    r"friday|fri|saturday|sat|sunday|sun"
)

# Words that mark the reminder as resolved/closed. Replying with any of these
# (optionally prefixed with the reminder verb) closes the reminder for good.
_CLOSE_WORDS = {
    "done", "completed", "complete", "received", "resolved",
    "cancel", "cancelled", "canceled", "stop", "close", "closed", "finished",
}
_CLOSE_RE = re.compile(
    rf"^\s*(?:please\s+)?(?:ok(?:ay)?[,!. ]+)?(?:{'|'.join(sorted(_CLOSE_WORDS))})\s*[.!?]*\s*$",
    re.IGNORECASE,
)

# Bare snooze interval, no "remind me" prefix and no subject:
#   "5m", "10 min", "30m", "1h", "2 hours", "in 1 hour", "1 day", "3 days"
_BARE_INTERVAL_RE = re.compile(
    rf"""
    ^\s*
    (?:please\s+)?
    (?:remind\s+me\s+)?
    (?:again\s+)?
    (?:in\s+)?
    (?P<num>{_NUMBER_PATTERN})
    \s*
    (?P<unit>{_UNIT_PATTERN})
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bare "tomorrow" / "tomorrow at 5:35am", no inline subject.
_BARE_TOMORROW_RE = re.compile(
    rf"""
    ^\s*
    (?:please\s+)?
    (?:remind\s+me\s+)?
    (?:again\s+)?
    tomorrow
    (?:\s+(?:at\s+)?{_CLOCK_RE})?
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Weekday, optionally with a clock time, no inline subject:
#   "Monday", "Monday 5:35am", "mon at 5:35", "Remind me Monday at 5:35am"
_BARE_WEEKDAY_RE = re.compile(
    rf"""
    ^\s*
    (?:please\s+)?
    (?:remind\s+me\s+)?
    (?:again\s+)?
    (?:on\s+|next\s+)?
    (?P<weekday>{_WEEKDAY_PATTERN})
    (?:\s+(?:at\s+)?{_CLOCK_RE})?
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bare "next week", no inline subject.
_BARE_NEXT_WEEK_RE = re.compile(
    r"^\s*(?:please\s+)?(?:remind\s+me\s+)?(?:again\s+)?next\s+week\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_REMINDER_BODY_RE = re.compile(
    r"(?:📅\s*)?REMINDER\s*:\s*(?P<body>.+)",
    re.IGNORECASE | re.DOTALL,
)

_PREFIX_LINE_RE = re.compile(r"^\s*(?:⚕\s*)?\*?Hermes Agent\*?\s*$", re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"^\s*[─\-—_]{3,}\s*$")
_REPLY_CONTEXT_RE = re.compile(r'^\s*\[Replying to:\s*".*"\]\s*$', re.IGNORECASE)


@dataclass(frozen=True)
class ParsedReminder:
    subject: str
    schedule: str
    display_time: str
    reused_reply_subject: bool = False


@dataclass(frozen=True)
class ParsedSnooze:
    """A reply-to-reminder reschedule/snooze action.

    ``subject`` is inherited from the quoted reminder. ``schedule`` is a string
    accepted by ``cron.jobs.parse_schedule`` (a relative duration like ``"1h"``
    / ``"7d"`` or an ISO timestamp for weekday/clock targets).
    """
    subject: str
    schedule: str
    display_time: str


@dataclass(frozen=True)
class ParsedReminderBatch:
    reminders: list[ParsedReminder]


def _parse_number(text: str) -> Optional[int]:
    value = (text or "").strip().lower()
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value)


def _clean_subject(text: str) -> str:
    subject = re.sub(r"\s+", " ", (text or "").strip())
    subject = subject.strip(" .!?:;,-")
    # Voice/STT often includes filler words before the subject.
    subject = re.sub(r"^(?:for|to|about)\s+", "", subject, flags=re.IGNORECASE).strip()
    return subject


def extract_subject_from_reminder_text(text: str | None) -> Optional[str]:
    """Extract the subject from a fired reminder message.

    Handles both the raw reminder shape and WhatsApp self-chat messages with the
    Hermes prefix prepended by the bridge.
    """
    if not text:
        return None
    cleaned_lines = []
    for line in str(text).splitlines():
        if _PREFIX_LINE_RE.match(line) or _SEPARATOR_RE.match(line) or _REPLY_CONTEXT_RE.match(line):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()

    match = _REMINDER_BODY_RE.search(cleaned)
    if not match:
        return None
    body = match.group("body").strip()
    # If another chunk or footer ever follows, keep only the first meaningful
    # non-empty line as the reminder subject.
    for line in body.splitlines():
        subject = _clean_subject(line)
        if subject:
            return subject
    return None


def extract_voice_transcript(text: str | None) -> Optional[str]:
    raw = str(text or "")
    transcripts = [m.group("text").strip() for m in _VOICE_TRANSCRIPT_RE.finditer(raw)]
    return transcripts[-1] if transcripts else None


def is_reminder_intent(text: str | None) -> bool:
    return bool(re.search(r"\bremind(?:er)?\b", str(text or ""), re.IGNORECASE))


def is_reminder_list_intent(text: str | None) -> bool:
    """Return True for obvious requests to list or show reminders."""
    body = str(text or "")
    return bool(
        re.search(
            r"\b(?:list|show|display|what(?:'s| is)?|give me|tell me)\b.*\breminders?\b|"
            r"\b(?:future|current|all)\b.*\breminders?\b|"
            r"\breminders?\b.*\b(?:on record|you have|current|future|all)\b",
            body,
            re.IGNORECASE,
        )
    )




def _candidate_lines(text: str) -> list[str]:
    """Return possible reminder command lines from newest to oldest.

    Voice messages reach the gateway wrapped as:
    ``[The user sent a voice message~ Here's what they said: "..."]``.
    Extract that transcript first so the deterministic reminder parser can run
    before the LLM sees the wrapper. For ordinary multi-line WhatsApp batches,
    try the last non-empty lines first to avoid carrying old reminder text into
    the current transaction.
    """
    raw = str(text or "").strip()
    if not raw:
        return []

    voice_candidates = []
    for match in _VOICE_TRANSCRIPT_RE.finditer(raw):
        transcript = match.group("text").strip()
        if transcript:
            voice_candidates.append(transcript)
    if voice_candidates:
        return list(reversed(voice_candidates))

    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not _REPLY_CONTEXT_RE.match(line.strip())
    ]
    return list(reversed(lines)) if lines else [raw]


def _parse_relative_line(line: str) -> Optional[re.Match[str]]:
    return (
        _RELATIVE_SUBJECT_AFTER_RE.match(line)
        or _RELATIVE_SUBJECT_AFTER_BARE_RE.match(line)
        or _RELATIVE_SUBJECT_BEFORE_RE.match(line)
    )


def _unit_to_schedule_and_display(unit_raw: str, number: int) -> tuple[str, str]:
    unit = unit_raw.lower()
    if unit.startswith(("h",)):
        return "h", "hour"
    if unit.startswith(("d",)):
        return "d", "day"
    if unit.startswith(("w",)):
        return f"{number * 7}d", "week"
    return "m", "minute"


def _parse_tomorrow_at_line(line: str, *, now: datetime | None = None) -> Optional[ParsedReminder]:
    match = _TOMORROW_AT_RE.match(line)
    if not match:
        return None

    hour = int(match.group("hour"))
    minute = int(match.group("minute") or "0")
    if hour < 1 or hour > 12 or minute < 0 or minute > 59:
        return None

    ampm = re.sub(r"[^apm]", "", match.group("ampm").lower())
    if ampm.startswith("p") and hour != 12:
        hour += 12
    elif ampm.startswith("a") and hour == 12:
        hour = 0

    subject = _clean_subject(match.group("subject"))
    if not subject:
        return None

    base = now or datetime.now().astimezone()
    run_at = (base + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    display_hour = run_at.strftime("%I").lstrip("0") or "0"
    display_time = f"Tomorrow {display_hour}:{run_at.strftime('%M')} {run_at.strftime('%p')}"
    return ParsedReminder(subject=subject, schedule=run_at.isoformat(), display_time=display_time)


def parse_simple_reminder_batch(
    text: str | None,
    *,
    reply_to_text: str | None = None,
    reply_to_subject_fallback: str | None = None,
    now: datetime | None = None,
) -> Optional[ParsedReminderBatch]:
    """Parse all high-confidence reminder requests in the current message.

    This deliberately ignores conversation history. Multi-line WhatsApp batches
    create one independent job per parsed line and return only those jobs in the
    confirmation.
    """
    reminders: list[ParsedReminder] = []
    candidates = list(reversed(_candidate_lines(text or "")))
    for line in candidates:
        parsed = _parse_tomorrow_at_line(line, now=now)
        if parsed is None:
            parsed = parse_simple_relative_reminder(
                line,
                reply_to_text=reply_to_text,
                reply_to_subject_fallback=reply_to_subject_fallback,
            )
        if parsed is not None:
            reminders.append(parsed)
        elif is_reminder_intent(line):
            # Mixed parsed/unparsed reminder commands are not safe for the
            # deterministic path. Fall back to the agent so it can clarify.
            return None
    if not reminders:
        return None
    return ParsedReminderBatch(reminders=reminders)


def parse_simple_relative_reminder(
    text: str | None,
    *,
    reply_to_text: str | None = None,
    reply_to_subject_fallback: str | None = None,
) -> Optional[ParsedReminder]:
    """Parse high-confidence relative reminder requests.

    Each parsed reminder is an independent transaction. The parser never
    consults prior reminders, active jobs, memory, or conversation history; the
    only external context allowed is the quoted reminder text for explicit
    WhatsApp replies like ``Remind me in 2 min``.
    """
    next_week_match = _NEXT_WEEK_RE.match(str(text or ""))
    if next_week_match:
        subject = _clean_subject(next_week_match.groupdict().get("subject") or "")
        reused_reply_subject = False
        if not subject:
            subject = extract_subject_from_reminder_text(reply_to_text) or _clean_subject(reply_to_subject_fallback or "")
            reused_reply_subject = bool(subject)
        if subject:
            return ParsedReminder(
                subject=subject,
                schedule="7d",
                display_time="in 1 week",
                reused_reply_subject=reused_reply_subject,
            )

    for line in _candidate_lines(text or ""):
        match = _parse_relative_line(line)
        if not match:
            continue

        number = _parse_number(match.group("num"))
        if not number or number <= 0:
            continue

        unit_raw = match.group("unit").lower()
        schedule_unit, unit_name = _unit_to_schedule_and_display(unit_raw, number)

        subject = _clean_subject(match.groupdict().get("subject") or "")
        if not subject:
            subject = extract_subject_from_reminder_text(reply_to_text) or _clean_subject(reply_to_subject_fallback or "")
        if not subject:
            continue
        reused_reply_subject = not bool(match.groupdict().get("subject") or "") and bool(
            reply_to_text or reply_to_subject_fallback
        )
        return ParsedReminder(
            subject=subject,
            schedule=f"{number}{schedule_unit}",
            display_time=f"in {number} {unit_name}{'' if number == 1 else 's'}",
            reused_reply_subject=reused_reply_subject,
        )


def is_reminder_close_intent(text: str | None) -> bool:
    """Return True when a reply means 'close/resolve this reminder'.

    Recognized words: done, completed, complete, received, resolved, cancel,
    cancelled, canceled, stop, close, closed, finished — optionally prefixed
    with 'ok'/'okay' or 'please'.
    """
    raw = str(text or "").strip()
    if not raw:
        return False
    # Voice replies arrive wrapped; unwrap the transcript first.
    transcript = extract_voice_transcript(raw)
    if transcript:
        raw = transcript.strip()
    return bool(_CLOSE_RE.match(raw))


def _next_weekday_run_at(weekday_idx: int, hour: int, minute: int, *, now: datetime) -> datetime:
    """Return the next datetime matching the given weekday and time.

    If today matches the weekday and the target time is still in the future,
    today is used; otherwise the next occurrence (1-7 days ahead) is chosen.
    """
    base = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    days_ahead = (weekday_idx - now.weekday()) % 7
    candidate = base + timedelta(days=days_ahead)
    if candidate <= now:
        candidate = candidate + timedelta(days=7)
    return candidate


def parse_snooze_reply(
    text: str | None,
    *,
    reply_to_text: str | None = None,
    reply_to_subject_fallback: str | None = None,
    now: datetime | None = None,
) -> Optional[ParsedSnooze]:
    """Parse a reply to a fired reminder as a reschedule/snooze action.

    The subject is ALWAYS inherited from the quoted reminder (``reply_to_text``)
    or the per-chat fallback — never from the reply body and never prompted for.
    This is the deterministic counterpart to the LLM's reply handling: a reply
    that names only a time means "move this reminder, keep its text".

    Returns ``None`` if the reply doesn't look like a time/interval, so the
    caller can fall through (e.g. to close-intent handling or the LLM).
    """
    raw = str(text or "").strip()
    if not raw:
        return None

    # Unwrap voice transcripts so "[... they said: \"monday 5:35am\"]" works.
    transcript = extract_voice_transcript(raw)
    if transcript:
        raw = transcript.strip()

    # Resolve the inherited subject. Without one, this is not a snooze we can
    # safely act on — bail so the caller can decide what to do.
    subject = extract_subject_from_reminder_text(reply_to_text) or _clean_subject(
        reply_to_subject_fallback or ""
    )
    if not subject:
        return None

    base_now = now or datetime.now().astimezone()

    # 1) Bare relative interval: "5m", "1h", "2 hours", "in 30 min", "3 days".
    m = _BARE_INTERVAL_RE.match(raw)
    if m:
        number = _parse_number(m.group("num"))
        if number and number > 0:
            schedule_unit, unit_name = _unit_to_schedule_and_display(m.group("unit").lower(), number)
            return ParsedSnooze(
                subject=subject,
                schedule=f"{number}{schedule_unit}",
                display_time=f"in {number} {unit_name}{'' if number == 1 else 's'}",
            )

    # 2) Bare "tomorrow" / "tomorrow at 5:35am".
    m = _BARE_TOMORROW_RE.match(raw)
    if m:
        hour, minute = _clock_from_match(m, default_hour=9, default_minute=0)
        if hour is not None:
            run_at = (base_now + timedelta(days=1)).replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            return ParsedSnooze(
                subject=subject,
                schedule=run_at.isoformat(),
                display_time=_display_dt(run_at, "Tomorrow"),
            )

    # 3) Weekday, optionally with a clock: "Monday", "Monday 5:35am".
    m = _BARE_WEEKDAY_RE.match(raw)
    if m:
        weekday_idx = _WEEKDAYS.get(m.group("weekday").lower())
        if weekday_idx is not None:
            hour, minute = _clock_from_match(m, default_hour=9, default_minute=0)
            if hour is not None:
                run_at = _next_weekday_run_at(weekday_idx, hour, minute, now=base_now)
                label = run_at.strftime("%A")
                return ParsedSnooze(
                    subject=subject,
                    schedule=run_at.isoformat(),
                    display_time=_display_dt(run_at, label),
                )

    # 4) Bare "next week".
    if _BARE_NEXT_WEEK_RE.match(raw):
        return ParsedSnooze(subject=subject, schedule="7d", display_time="in 1 week")

    return None


def _clock_from_match(match: re.Match[str], *, default_hour: int, default_minute: int) -> tuple[Optional[int], int]:
    """Extract (hour_24, minute) from a regex match exposing the _CLOCK_RE groups.

    Returns (default_hour, default_minute) when no clock was supplied, or
    (None, 0) when the clock is present but invalid.
    """
    groups = match.groupdict()
    if not groups.get("hour"):
        return default_hour, default_minute
    hour = int(groups["hour"])
    minute = int(groups.get("minute") or "0")
    if hour < 1 or hour > 12 or minute < 0 or minute > 59:
        return None, 0
    ampm = re.sub(r"[^apm]", "", (groups.get("ampm") or "").lower())
    if ampm.startswith("p") and hour != 12:
        hour += 12
    elif ampm.startswith("a") and hour == 12:
        hour = 0
    elif not ampm:
        # No am/pm given (e.g. "Monday 5:35"). Keep the literal hour; this is a
        # best-effort interpretation for a control-plane reschedule.
        pass
    return hour, minute


def _display_dt(run_at: datetime, label: str) -> str:
    display_hour = run_at.strftime("%I").lstrip("0") or "0"
    return f"{label} {display_hour}:{run_at.strftime('%M')} {run_at.strftime('%p')}"


def reminder_prompt(subject: str) -> str:
    return f"Remind the user: {_clean_subject(subject)}"


def reminder_confirmation(parsed: ParsedReminder) -> str:
    if parsed.reused_reply_subject:
        return f"Done — reusing quoted reminder: {parsed.subject} {parsed.display_time}."
    return f"Done — I set a reminder for {parsed.subject} {parsed.display_time}."


def reminder_output(subject: str) -> str:
    return f"📅 REMINDER:\n{_clean_subject(subject)}"


# --- Persistent per-chat last-reminder-subject store ---------------------------
#
# Defense-in-depth for Change A: when a reminder *fires* and is delivered to a
# chat, we record its subject keyed by chat_id in a small JSON file. If the user
# later replies and the platform stripped the quoted reminder text, the reply
# path can still recover the subject from here. This survives the cron-thread
# boundary and gateway restarts (the in-memory cache in run.py does not).

_SUBJECT_STORE_TTL_SECONDS = 7 * 24 * 3600  # a reminder chain can span a week


def _subject_store_path():
    from pathlib import Path
    import os
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    return Path(home) / "reminder_last_subject.json"


def record_last_reminder_subject(chat_id: str | None, subject: str | None, *, now_ts: float | None = None) -> None:
    """Persist the last reminder subject delivered to ``chat_id``.

    Best-effort: never raises into the delivery path.
    """
    if not chat_id or not subject:
        return
    import json
    import time as _time
    ts = now_ts if now_ts is not None else _time.time()
    path = _subject_store_path()
    try:
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text()) or {}
            except Exception:
                data = {}
        # Prune stale entries while we're here.
        cutoff = ts - _SUBJECT_STORE_TTL_SECONDS
        data = {
            k: v for k, v in data.items()
            if isinstance(v, dict) and float(v.get("ts", 0)) >= cutoff
        }
        data[str(chat_id)] = {"subject": _clean_subject(subject), "ts": ts}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
    except Exception:
        # Persistence is a fallback, not a hard dependency.
        pass


def lookup_last_reminder_subject(chat_id: str | None, *, now_ts: float | None = None) -> Optional[str]:
    """Return the most recently delivered reminder subject for ``chat_id``."""
    if not chat_id:
        return None
    import json
    import time as _time
    ts = now_ts if now_ts is not None else _time.time()
    path = _subject_store_path()
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text()) or {}
        entry = data.get(str(chat_id))
        if not isinstance(entry, dict):
            return None
        if float(entry.get("ts", 0)) < ts - _SUBJECT_STORE_TTL_SECONDS:
            return None
        return entry.get("subject") or None
    except Exception:
        return None


def clear_last_reminder_subject(chat_id: str | None) -> None:
    """Remove the stored subject for ``chat_id`` (e.g. after the reminder is closed)."""
    if not chat_id:
        return
    import json
    path = _subject_store_path()
    try:
        if not path.exists():
            return
        data = json.loads(path.read_text()) or {}
        if str(chat_id) in data:
            del data[str(chat_id)]
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
    except Exception:
        pass


def _reminder_subject_from_job(job: dict) -> str:
    subject = extract_subject_from_reminder_text(job.get("prompt"))
    if subject:
        return subject
    name = _clean_subject(job.get("name") or "")
    if name.lower().startswith("reminder:"):
        name = name.split(":", 1)[1].strip()
    return name or _clean_subject(job.get("prompt") or "") or "(untitled)"


def reminder_list_output(jobs: list[dict]) -> str:
    """Format reminder jobs for direct chat delivery."""
    filtered = [j for j in jobs if j]
    if not filtered:
        return "No future reminders on record."

    def _sort_key(job: dict):
        return str(job.get("next_run_at") or "~"), str(job.get("schedule_display") or ""), str(job.get("name") or "")

    lines = ["Here are your future reminders:"]
    for job in sorted(filtered, key=_sort_key):
        schedule = _clean_subject(job.get("schedule_display") or job.get("schedule") or "?")
        schedule_kind = str((job.get("schedule") or {}).get("kind") or "").lower()
        repeat_times = (job.get("repeat") or {}).get("times")
        kind = "Recurring" if schedule_kind in {"cron", "interval"} or repeat_times is None else "Once"
        subject = _reminder_subject_from_job(job)
        lines.append(f"{kind} — {schedule} — {subject}")
    return "\n".join(lines)
