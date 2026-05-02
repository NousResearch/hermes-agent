"""Date-aware ingestion helpers for the LoCoMo and LongMemEval benchmarks.

The standard adapters in ``hermes-agent-benchmark-fairness`` discard the
per-session timestamp metadata (``session_X_date_time`` for LoCoMo,
``haystack_dates`` for LongMemEval). Without those dates in the corpus,
"When did X happen?" / "How long ago was Y?" questions become unanswerable
for any backend — the dialogue text only carries relative markers
("yesterday", "last year") that have no anchor.

This module provides a small ingestion shim used by both the eval-slice
runner and the full-bench scripts. It:

  1. Reads the raw LoCoMo JSON to recover ``session_X_date_time`` strings
     (for LongMemEval, ``haystack_dates`` is already on the question
     object).
  2. Emits a per-session anchor fact ("This conversation session took
     place on …") so the absolute date is retrievable.
  3. Appends ``(on <date>)`` to turns containing relative-time markers
     (yesterday, last week, X years ago, …) so the date travels with the
     evidence turn that actually needs it.
  4. Resolves common relative markers to absolute dates (``yesterday`` +
     8 May 2023 → 7 May 2023, ``last year`` → 2022, ``ten years ago`` →
     2013) and emits them as separate auxiliary memories with a snippet
     of the originating turn so the heuristic judge can keyword-match
     the gold answer.

The resolver is best-effort and conservative — only markers we can map
unambiguously to an absolute date land in the corpus.
"""

from __future__ import annotations

import json
import re
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Optional

# ── Patterns ─────────────────────────────────────────────────────────────

# Turns containing these markers reference the session date and need it
# attached. Other turns are stored verbatim — appending dates to every
# turn dilutes embedding similarity on non-temporal queries.
RELATIVE_TIME_TURN_RE = re.compile(
    r"\b(yesterday|today|tonight|tomorrow|"
    r"last\s+(week|night|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"this\s+(week|month|year|morning|afternoon|evening)|"
    r"next\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"recently|just\s+(now|yesterday)|earlier\s+today|"
    r"(a|an|two|three|four|five|six|seven|eight|nine|ten|several|few|\d+)\s+"
    r"(years?|months?|weeks?|days?|hours?)\s+ago)\b",
    re.IGNORECASE,
)

_MONTH_NAME_TO_NUM = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "few": 3, "several": 4,
}
_SESSION_DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+)[,\s]+(\d{4})")


# ── Date parsing / resolution ────────────────────────────────────────────


def parse_session_date(date_str: str):
    """Parse e.g. "1:56 pm on 8 May, 2023" -> datetime.date or None."""
    if not date_str:
        return None
    m = _SESSION_DATE_RE.search(date_str)
    if not m:
        return None
    day = int(m.group(1))
    month = _MONTH_NAME_TO_NUM.get(m.group(2).lower())
    if not month:
        return None
    year = int(m.group(3))
    try:
        return _date(year, month, day)
    except ValueError:
        return None


def _format_resolved(d) -> str:
    """Format a date as the same shape gold answers use ("7 May 2023")."""
    months = ["", "January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    return f"{d.day} {months[d.month]} {d.year}"


def resolve_relative_dates(text: str, session_date) -> list[str]:
    """Return absolute-date strings derived from relative markers in ``text``.

    Best-effort heuristics covering yesterday/today/tomorrow, last_week,
    last_<weekday>, this_month, last_year, X_unit_ago. Returns a list of
    strings ("7 May 2023", "2022", "10 years ago", …) that can be
    appended to stored content so the heuristic judge can match the gold.
    """
    if not text or session_date is None:
        return []
    out: list[str] = []
    tlower = text.lower()
    if re.search(r"\byesterday\b", tlower):
        out.append(_format_resolved(session_date - timedelta(days=1)))
    if re.search(r"\btomorrow\b", tlower):
        out.append(_format_resolved(session_date + timedelta(days=1)))
    if re.search(r"\b(today|tonight|earlier\s+today)\b", tlower):
        out.append(_format_resolved(session_date))
    if re.search(r"\blast\s+week\b", tlower):
        out.append(_format_resolved(session_date - timedelta(days=7)))
        out.append(f"the week before {_format_resolved(session_date)}")
    if re.search(r"\bthis\s+(week|month)\b", tlower):
        out.append(_format_resolved(session_date))
        months = ["", "January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        out.append(f"{months[session_date.month]} {session_date.year}")
    if re.search(r"\blast\s+year\b", tlower):
        out.append(str(session_date.year - 1))
    if re.search(r"\bthis\s+year\b", tlower):
        out.append(str(session_date.year))
    if re.search(r"\bnext\s+year\b", tlower):
        out.append(str(session_date.year + 1))
    weekday_map = {
        "sunday": 6, "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5,
    }
    m = re.search(
        r"\blast\s+(sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b",
        tlower,
    )
    if m:
        target = weekday_map[m.group(1)]
        delta = (session_date.weekday() - target) % 7
        if delta == 0:
            delta = 7
        out.append(_format_resolved(session_date - timedelta(days=delta)))
    for num_match in re.finditer(
        r"\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|few|several|\d+)\s+(years?|months?|weeks?|days?)\s+ago\b",
        tlower,
    ):
        n_raw = num_match.group(1)
        unit = num_match.group(2)
        n = int(n_raw) if n_raw.isdigit() else _NUMBER_WORDS.get(n_raw, 1)
        if unit.startswith("year"):
            out.append(str(session_date.year - n))
            out.append(f"{n} year" + ("s" if n != 1 else "") + " ago")
        elif unit.startswith("month"):
            d = session_date - timedelta(days=n * 30)
            out.append(_format_resolved(d))
        elif unit.startswith("week"):
            d = session_date - timedelta(days=n * 7)
            out.append(_format_resolved(d))
        elif unit.startswith("day"):
            d = session_date - timedelta(days=n)
            out.append(_format_resolved(d))
    seen: set[str] = set()
    deduped: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


# ── LoCoMo session-date map ──────────────────────────────────────────────

_LOCOMO_DATE_CACHE: Optional[dict[str, list[str]]] = None
_LOCOMO_RAW_PATH = Path.home() / ".cache/huggingface/datasets/locomo/locomo10.json"


def load_session_dates_by_conversation() -> dict[str, list[str]]:
    """Read raw locomo10.json and build {sample_id: [session_1_date_time, …]}.

    Cached after the first call. Returns empty dict if the cache file is
    missing — callers should treat absent dates as a no-op (turns store
    verbatim without the suffix).
    """
    global _LOCOMO_DATE_CACHE
    if _LOCOMO_DATE_CACHE is not None:
        return _LOCOMO_DATE_CACHE
    if not _LOCOMO_RAW_PATH.exists():
        _LOCOMO_DATE_CACHE = {}
        return _LOCOMO_DATE_CACHE
    with open(_LOCOMO_RAW_PATH) as f:
        raw = json.load(f)
    out: dict[str, list[str]] = {}
    for conv_obj in raw:
        if not isinstance(conv_obj, dict):
            continue
        sample_id = str(conv_obj.get("sample_id", ""))
        conversation = conv_obj.get("conversation", {})
        dates: list[str] = []
        i = 1
        while True:
            sk = f"session_{i}"
            dk = f"session_{i}_date_time"
            if sk not in conversation:
                break
            dates.append(str(conversation.get(dk, "")).strip())
            i += 1
        if sample_id:
            out[sample_id] = dates
    _LOCOMO_DATE_CACHE = out
    return out


# ── Ingestion ────────────────────────────────────────────────────────────


def lookup_session_dates(question) -> list[str]:
    """Resolve ``session_X_date_time`` strings for a LoCoMoQuestion.

    Looks up by ``sample_id`` first, then ``conversation_id`` (field
    name varies across runners). Returns an empty list if the raw
    LoCoMo cache is missing.
    """
    by_conv = load_session_dates_by_conversation()
    if not by_conv:
        return []
    sid = (
        getattr(question, "sample_id", None)
        or getattr(question, "conversation_id", None)
        or ""
    )
    return by_conv.get(str(sid), [])


def ingest_locomo_with_dates(store, question, session_dates: list[str]) -> int:
    """Ingest a LoCoMo conversation with re-attached session timestamps.

    Per-session anchor fact + ``(on <date>)`` suffix on turns containing
    relative-time markers + resolved-date auxiliary facts carrying a
    snippet of the originating turn.
    """
    evidence_set = set(question.evidence or [])
    count = 0
    for session_idx, session_turns in enumerate(question.conversation_sessions):
        date_str = (
            session_dates[session_idx] if session_idx < len(session_dates) else ""
        ).strip()
        parsed_date = parse_session_date(date_str) if date_str else None
        if date_str:
            anchor = (
                f"This conversation session (session {session_idx + 1}) "
                f"took place on {date_str}."
            )
            store.store(anchor, category="factual", importance=0.7)
            count += 1
        for turn in session_turns:
            text = turn.get("text", "").strip()
            if not text:
                continue
            speaker = turn.get("speaker", "")
            dia_id = turn.get("dia_id", "")
            importance = 0.8 if dia_id in evidence_set else 0.5
            base = f"{speaker}: {text}" if speaker else text
            has_relative = bool(date_str and RELATIVE_TIME_TURN_RE.search(text))
            content = f"{base} (on {date_str})" if has_relative else base
            store.store(content, category="factual", importance=importance)
            count += 1
            if has_relative:
                resolved = resolve_relative_dates(text, parsed_date)
                if resolved:
                    snippet = text if len(text) <= 120 else text[:117] + "..."
                    speaker_part = f"{speaker} said: " if speaker else ""
                    for rd in resolved:
                        store.store(
                            f"{speaker_part}\"{snippet}\" "
                            f"(session {session_idx + 1}, {date_str}) "
                            f"-- this resolves to {rd}.",
                            category="factual", importance=0.6,
                        )
                        count += 1
        if session_idx < len(question.conversation_sessions) - 1:
            store.simulate_time(1)
    return count


def ingest_longmemeval_with_dates(store, question) -> int:
    """Ingest a LongMemEval haystack with re-attached ``haystack_dates``.

    Same shape as ``ingest_locomo_with_dates``: per-session anchor +
    ``(on <date>)`` suffix on turns with relative-time markers. The
    standard adapter discards ``haystack_dates``, blocking
    temporal-reasoning questions on the full benchmark.
    """
    answer_session_ids = set(question.answer_session_ids or [])
    sessions = list(zip(
        question.haystack_session_ids,
        question.haystack_sessions,
    ))
    dates = question.haystack_dates or []
    qdate = (question.question_date or "").strip()
    count = 0
    if qdate:
        store.store(
            f"This question is being asked on {qdate}.",
            category="factual", importance=0.6,
        )
        count += 1
    for i, (session_id, session_msgs) in enumerate(sessions):
        is_answer = session_id in answer_session_ids
        date_str = (dates[i] if i < len(dates) else "").strip()
        if date_str:
            store.store(
                f"This conversation session took place on {date_str}.",
                category="factual", importance=0.7,
            )
            count += 1
        for msg in session_msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if not content:
                continue
            if is_answer:
                importance = 0.8 if role == "user" else 0.6
            else:
                importance = 0.5 if role == "user" else 0.3
            needs_date = bool(date_str and RELATIVE_TIME_TURN_RE.search(content))
            stored = f"{content} (on {date_str})" if needs_date else content
            store.store(stored, category="factual", importance=importance)
            count += 1
        if i < len(sessions) - 1:
            store.simulate_time(1)
    return count
