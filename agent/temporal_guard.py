"""Temporal-claim guard: verify the model's clock claims against the stamp.

Companion to the per-turn timestamp feature.  ``run_conversation`` stamps
the authoritative wall clock (``agent._current_turn_timestamp``) into every
request-only system message; this module checks the assistant's FINAL
response text for temporal claims that contradict that stamp, so a model
that pattern-matches a plausible time instead of reading the stamp shows up
in the logs instead of silently misleading the user.

Scope (deliberately narrow to keep false positives near zero):

- **now-claims**: explicit "it's <HH:MM>" / "the time is <HH:MM>" phrasing,
  compared against the stamp within ``tolerance_min``.
- **anchored recency**: a clock time and a "N minutes/hours ago" phrase in
  the SAME sentence, where the arithmetic must match (stamp - clock).

Deliberately NOT checked (documented blind spots):

- Unanchored recency ("about 15 minutes ago" with no clock time in the
  sentence) — there is no reference point to verify against.
- Event references ("the log at 14:42 shows", "started 23:46:45 last
  night") — past events, not now-claims.
- Future/schedule claims ("the watcher fires at 06:45").
- Times inside fenced/inline code, location-qualified times ("it's 9am in
  Tokyo"), version numbers, and ISO timestamps.

The guard is log-only: it never mutates the response and never raises.
"""

from __future__ import annotations

import re

_STAMP_RE = re.compile(r"Current time: (\w+) (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) (\w+)")
_NOW_CLAIM_RE = re.compile(
    r"(?:it'?s|now|currently|right now)\s+(?:about|around|roughly|~)?\s*"
    r"(\d{1,2}):(\d{2})(?:\s*(am|pm))?\b",
    re.IGNORECASE,
)
_TIME_IS_RE = re.compile(
    r"(?:current time|the time|time)\s+is\s+(?:about|around|roughly|~)?\s*"
    r"(\d{1,2}):(\d{2})(?:\s*(am|pm))?\b",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(Z|[A-Z]{2,4})?\b")
_AGO_RE = re.compile(r"\b(\d+)\s*(min(?:ute)?s?|hour(?:s)?|hrs?|h)\s+ago\b", re.IGNORECASE)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")


def _to_minutes(h: str, m: str, ap: str | None = None) -> int:
    hh = int(h) % 24
    if ap:
        ap = ap.lower()
        if ap == "pm" and hh < 12:
            hh += 12
        if ap == "am" and hh == 12:
            hh = 0
    return hh * 60 + int(m)


def _wrap_diff(a_min: int, b_min: int) -> int:
    """Circular-clock distance, so 23:58 vs 00:05 is 7 minutes, not 1433."""
    raw = abs(a_min - b_min)
    return min(raw, 24 * 60 - raw)


def check_temporal_claims(text: str, stamp: str, tolerance_min: int = 5) -> list[str]:
    """Return human-readable flags for temporal claims contradicting ``stamp``.

    Empty text, an unparseable stamp, or no contradicting claim all return
    ``[]``. The check is best-effort; callers should wrap it in try/except
    and never block a turn on its result.
    """
    flags: list[str] = []
    if not text or not stamp:
        return flags
    m = _STAMP_RE.search(stamp)
    if not m:
        return flags
    shh, smm = m.group(3).split(":")
    stamp_min = int(shh) * 60 + int(smm)

    # Strip code before matching: literal times inside code are not claims.
    scan_text = _FENCE_RE.sub(" ", text)
    scan_text = _INLINE_CODE_RE.sub(" ", scan_text)

    # 1. now-claims
    for rx in (_NOW_CLAIM_RE, _TIME_IS_RE):
        for cm in rx.finditer(scan_text):
            tail = scan_text[cm.end() : cm.end() + 30]
            if re.match(r"\s+(?:in|at)\s+[A-Za-z]", tail):
                continue  # "it's 9am in Tokyo" — location/timezone, not local now
            claim_min = _to_minutes(cm.group(1), cm.group(2), cm.group(3))
            if _wrap_diff(claim_min, stamp_min) > tolerance_min:
                flags.append(
                    f"now-claim {cm.group(0)!r} contradicts stamp "
                    f"{stamp_min // 60:02d}:{stamp_min % 60:02d}"
                )

    # 2. anchored recency: clock time + "N ... ago" in the same sentence
    for sent in re.split(r"(?<=[.!?])\s+|\n", scan_text):
        times = [(tm, _to_minutes(tm.group(1), tm.group(2))) for tm in _CLOCK_RE.finditer(sent)]
        agos = [(am, int(am.group(1)), am.group(2).lower()) for am in _AGO_RE.finditer(sent)]
        if not times or not agos:
            continue
        for tm, tmin in times:
            for am, n, unit in agos:
                claimed = n * (60 if unit.startswith("h") else 1)
                expected = (stamp_min - tmin) % (24 * 60)
                if expected > 12 * 60:
                    continue  # clock wrap: ambiguous, do not guess
                if abs(expected - claimed) > max(10, claimed * 0.25):
                    flags.append(
                        f"recency {am.group(0)!r} at clock {tm.group(0)!r}: "
                        f"expected {expected}min vs claimed {claimed}min"
                    )
    return flags
