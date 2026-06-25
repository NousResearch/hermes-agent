"""Plugin-side temporal-expression parsing for Mem0 recall (W3-TEMPORAL).

Mirrors Hindsight's recency subset (spec §2.1): detect a temporal expression in a
query, resolve it to a ``created_at [start, end)`` UTC window, and hand it to the
recall path as a **filter+boost** over the over-fetched semantic candidate pool.

This is the τ_m (mention/learned time) axis only — ``created_at`` is the time we
learned a fact, which the W3 pre-req probe confirmed is mention-time-stable
(update preserves it; supersession stamps only ``superseded_by``). Bi-temporal
event-time (τ_s/τ_e) is deliberately deferred.

Why plugin-side, not a server ``created_at`` range filter: the self-hosted
``_build_filter_conditions`` only exposes ``gte``/``lte`` with a ``::numeric``
cast — it cannot express an ISO-8601 timestamp text-range — and the server-side
``reference_date`` path raises in sync ``search``. So the window is resolved here
and applied as a client-side re-rank/filter on the results ``/search`` already
returns (each result carries ``created_at``). No server change, no image rebuild.

Date math is the proven DST-correct PT-day → UTC bounds from the digest work:
a day ``d`` in the reference zone maps to
``[ tz_midnight(d) → UTC, tz_midnight(d+1) → UTC )`` via ``zoneinfo``, so
spring-forward days are 23h, fall-back days 25h, and an exact-midnight instant
lands in-day on the start side (half-open ``[start, end)``).

Stdlib only (re + datetime + zoneinfo) — no new plugin dependency.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

# The reference zone for "a day". Ace's agent lives on Pacific time and the digest
# bounds are PT-day; this keeps "the 20th" meaning the human PT calendar day.
DEFAULT_TZ = "America/Los_Angeles"

UTCWindow = Tuple[datetime, datetime]  # [start, end) — both tz-aware UTC

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

# Explicit ISO date: 2026-06-20
_RE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# "<Month> <day>" e.g. "June 20", "June 20th", "Jun 20"
_RE_MONTH_DAY = re.compile(
    r"\b(" + _MONTH_ALT + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.IGNORECASE)
# "<day> of <Month>" e.g. "20th of June", "21 of June"
_RE_DAY_OF_MONTH = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+of\s+(" + _MONTH_ALT + r")\b", re.IGNORECASE)
# "mid-June" / "mid June"
_RE_MID_MONTH = re.compile(r"\bmid[-\s]+(" + _MONTH_ALT + r")\b", re.IGNORECASE)
# "in June" / "during June" (whole month)
_RE_IN_MONTH = re.compile(
    r"\b(?:in|during|throughout)\s+(" + _MONTH_ALT + r")\b", re.IGNORECASE)
# bare "the 20th" / "on the 21st" (day-of-month, month inferred from reference).
# Negative lookahead guards against ordinal-as-rank phrasing ("the 3rd result",
# "the 1st warning", "the 2nd item") — those are NOT day-of-month references and
# would otherwise silently promote memories from that calendar day. The excluded
# nouns are the common "Nth <thing>" enumerators; a real date use ("on the 21st",
# "the 21st we shipped") doesn't put one of these immediately after the ordinal.
_RE_THE_NTH = re.compile(
    r"\b(?:on\s+)?the\s+(\d{1,2})(?:st|nd|rd|th)\b"
    r"(?!\s+(?:result|item|one|time|place|step|warning|entry|entries|row|line|"
    r"example|option|answer|attempt|try|version|part|point|section|paragraph|"
    r"page|chapter|message|reply|replies|note|record|element|column|field|"
    r"results|items|times|places|steps|warnings|rows|lines|options|answers|"
    r"attempts|versions|parts|points|sections|paragraphs|pages|chapters|"
    r"messages|notes|records|elements|columns|fields)\b)",
    re.IGNORECASE,
)
# relative
_RE_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)
_RE_TODAY = re.compile(r"\btoday\b", re.IGNORECASE)
_RE_LAST_WEEK = re.compile(r"\blast\s+week\b", re.IGNORECASE)
_RE_LAST_MONTH = re.compile(r"\blast\s+month\b", re.IGNORECASE)
_RE_PAST_N_DAYS = re.compile(r"\b(?:past|last)\s+(\d{1,2})\s+days?\b", re.IGNORECASE)


def _day_window(d: date, tz: ZoneInfo) -> UTCWindow:
    """[start, end) UTC for the reference-zone calendar day ``d``.

    DST-correct: maps tz-midnight(d) and tz-midnight(d+1) to UTC, so a
    spring-forward day spans 23h and a fall-back day 25h. Half-open: an instant
    at exactly tz-midnight lands in-day; tz-midnight(d+1) is the first out-of-day
    instant.
    """
    # Anchor both bounds to tz-midnight of the calendar day (NOT aware-datetime
    # + 24h, which is wall-clock 24h and the wrong instant across a DST fold).
    nxt = d + timedelta(days=1)
    start_local = datetime(d.year, d.month, d.day, tzinfo=tz)
    end_local = datetime(nxt.year, nxt.month, nxt.day, tzinfo=tz)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _span_window(d_start: date, d_end_inclusive: date, tz: ZoneInfo) -> UTCWindow:
    """[start, end) UTC spanning the reference-zone days ``d_start``..``d_end_inclusive``."""
    start, _ = _day_window(d_start, tz)
    _, end = _day_window(d_end_inclusive, tz)
    return start, end


def _clamp_day(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_temporal_window(
    query: str,
    *,
    reference_date: Optional[date] = None,
    tz_name: str = DEFAULT_TZ,
) -> Optional[UTCWindow]:
    """Resolve the first temporal expression in ``query`` to a ``[start, end)`` UTC window.

    Returns None when no temporal expression is detected (the recall path then
    behaves exactly as before — no window, no boost).

    Precedence (most-specific / explicit first): explicit ISO date → "<Month>
    <day>" / "<day> of <Month>" → "the <N>th" → "mid-<Month>" → "in <Month>" →
    yesterday/today → past/last N days → last week → last month. Explicit date
    tokens win over relative words, so e.g. "yesterday's tests on the 21st"
    resolves to the 21st, not yesterday.

    ``reference_date`` anchors relative/bare-day expressions (defaults to today
    in the reference zone). ``tz_name`` is the calendar-day zone (default PT).
    """
    if not query:
        return None
    tz = ZoneInfo(tz_name)
    if reference_date is None:
        reference_date = datetime.now(tz).date()
    ref = reference_date

    # 1. explicit ISO date
    m = _RE_ISO.search(query)
    if m:
        d = _clamp_day(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            return _day_window(d, tz)

    # 2. "<Month> <day>" or "<day> of <Month>"
    for rex, mo_idx, day_idx in ((_RE_MONTH_DAY, 1, 2), (_RE_DAY_OF_MONTH, 2, 1)):
        m = rex.search(query)
        if m:
            month = _MONTHS[m.group(mo_idx).lower()]
            day = int(m.group(day_idx))
            year = ref.year
            # a future month/day under the same year is most likely last year's mention
            d = _clamp_day(year, month, day)
            if d and d > ref:
                d = _clamp_day(year - 1, month, day)
            if d:
                return _day_window(d, tz)

    # 3. bare "the <N>th" — infer month/year from reference (most recent past day-of-month)
    m = _RE_THE_NTH.search(query)
    if m:
        day = int(m.group(1))
        d = _clamp_day(ref.year, ref.month, day)
        if d and d > ref:
            # day-of-month hasn't happened yet this month → previous month
            pm_year, pm_month = (ref.year, ref.month - 1) if ref.month > 1 else (ref.year - 1, 12)
            d = _clamp_day(pm_year, pm_month, day)
        if d:
            return _day_window(d, tz)

    # 4. "mid-<Month>" — the middle third (days 11..20)
    m = _RE_MID_MONTH.search(query)
    if m:
        month = _MONTHS[m.group(1).lower()]
        year = ref.year
        if _clamp_day(year, month, 1) and date(year, month, 1) > ref:
            year -= 1
        start = _clamp_day(year, month, 11)
        end = _clamp_day(year, month, 20)
        if start and end:
            return _span_window(start, end, tz)

    # 5. "in <Month>" — the whole month
    m = _RE_IN_MONTH.search(query)
    if m:
        month = _MONTHS[m.group(1).lower()]
        year = ref.year
        if date(year, month, 1) > ref:
            year -= 1
        first = date(year, month, 1)
        last = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
        return _span_window(first, last, tz)

    # 6. yesterday / today
    if _RE_YESTERDAY.search(query):
        return _day_window(ref - timedelta(days=1), tz)
    if _RE_TODAY.search(query):
        return _day_window(ref, tz)

    # 7. "past/last N days" — the prior N full days (excludes today)
    m = _RE_PAST_N_DAYS.search(query)
    if m:
        n = int(m.group(1))
        if n >= 1:
            return _span_window(ref - timedelta(days=n), ref - timedelta(days=1), tz)

    # 8. "last week" — the prior 7 full days (excludes today)
    if _RE_LAST_WEEK.search(query):
        return _span_window(ref - timedelta(days=7), ref - timedelta(days=1), tz)

    # 9. "last month" — the previous calendar month
    if _RE_LAST_MONTH.search(query):
        pm_year, pm_month = (ref.year, ref.month - 1) if ref.month > 1 else (ref.year - 1, 12)
        first = date(pm_year, pm_month, 1)
        last = date(ref.year, ref.month, 1) - timedelta(days=1)
        return _span_window(first, last, tz)

    return None


def created_at_in_window(created_at: Optional[str], window: UTCWindow) -> bool:
    """True if an ISO-8601 ``created_at`` string falls in the half-open UTC window.

    Tolerant of the store's ``+00:00`` / ``-00:00`` / ``Z`` suffixes and a missing
    offset (treated as UTC). Unparseable / missing → False (out of window).
    """
    if not created_at:
        return False
    s = created_at.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    start, end = window
    return start <= dt < end
