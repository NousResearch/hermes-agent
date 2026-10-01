"""Retry utilities — jittered backoff for decorrelated retries.

Jittered delays (vs. fixed exponential) prevent thundering-herd retry spikes
when many sessions hit the same rate-limited provider concurrently.
"""

import random
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

# Monotonic counter for jitter-seed uniqueness within a process; locked
# because concurrent gateway sessions retry simultaneously.
_jitter_counter = 0
_jitter_lock = threading.Lock()

# Z.AI Coding Plan's GLM-5.2 endpoint often returns 429 code 1305 ("service may be
# temporarily overloaded"). Short retries hammer the same window, so after
# ``_ZAI_CODING_OVERLOAD_SHORT_ATTEMPTS`` normal retries the wait widens progressively;
# the cap stays interactive-friendly (a TUI message should fail visibly in minutes).
# The short count is shared by ``adaptive_rate_limit_backoff`` and
# ``zai_coding_overload_retry_ceiling`` so the two cannot silently desync.
_ZAI_CODING_OVERLOAD_LONG_BACKOFF = (30.0, 60.0, 90.0, 120.0)
_ZAI_CODING_OVERLOAD_SHORT_ATTEMPTS = 3


def parse_retry_after_seconds(value_or_headers: Any) -> Optional[float]:
    """Parse a ``Retry-After`` value (numeric / HTTP-date) or a headers mapping (both casings tried) into
    seconds, clamped at 0.0; None when absent / unparseable."""
    raw = value_or_headers
    if raw is not None and not isinstance(raw, (str, int, float)):
        getter = getattr(raw, "get", None)
        if not callable(getter):
            return None
        try:
            raw = getter("Retry-After")
            if raw is None:
                raw = getter("retry-after")
        except Exception:
            return None
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return max(0.0, float(raw))
    text = str(raw).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except (TypeError, ValueError):
        pass
    # HTTP-date form (RFC 7231): seconds until that instant, clamped at 0.
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:  # older stdlib returns None instead of raising
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


# Free-text "reset" grammars providers put in error bodies, tried in order. One table so the
# conversation loop's error context and the credential pool's cooldown agree on the same wait.
_QUOTA_RESET_DELAY_RE = re.compile(r"quotaResetDelay[:\s\"]+(\d+(?:\.\d+)?)(ms|s)", re.IGNORECASE)
# "Resets in 4hr 5min" (weekly usage limits), "resets in 2 hours 5 minutes", "resets in 30s".
_RESETS_IN_RE = re.compile(
    r"resets?\s+in\s+"
    r"(?:(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b\s*)?"
    r"(?:(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b\s*)?"
    r"(?:(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b)?", re.IGNORECASE,
)
_RETRY_AFTER_SECONDS_RE = re.compile(r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)", re.IGNORECASE)
# The plan usage-limit body field as it appears once stringified: ``'resets_in_seconds': 30995``.
_RESETS_IN_SECONDS_FIELD_RE = re.compile(r"resets_in_seconds\W{1,4}(\d+(?:\.\d+)?)", re.IGNORECASE)
# Wall-clock reset grammar: "resets 12:30pm (America/Chicago)" — Claude subscription session/
# weekly limits name the reopen as a time in an EXPLICIT zone, not a delta. The zone is
# required (an unnamed "resets 5pm" is someone's local clock; guessing can be hours wrong)
# and unknown zones fail open to the pre-existing behavior. Interpreted as the NEXT
# occurrence of that wall-clock time, with a just-past tolerance: an error that surfaces a
# minute after the stated instant must not inflate the wait by a whole day.
_RESETS_CLOCK_RE = re.compile(
    r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*\(([^()\s]{3,64})\)",
    re.IGNORECASE,
)
_RESET_CLOCK_JUST_PAST_TOLERANCE_S = 30 * 60.0
_RESET_CLOCK_MAX_S = 24 * 3600.0


def _quota_reset_seconds(m: "re.Match[str]") -> float:
    value = float(m.group(1))
    return value / 1000.0 if m.group(2).lower() == "ms" else value


def _resets_in_seconds(m: "re.Match[str]") -> Optional[float]:
    if not any(m.groups()):  # "resets in" with no unit-bearing number: not this grammar
        return None
    return float(m.group(1) or 0) * 3600 + float(m.group(2) or 0) * 60 + float(m.group(3) or 0)


def _resets_clock_seconds(m: "re.Match[str]") -> Optional[float]:
    """Seconds until the NEXT occurrence of the wall-clock reset named in an explicit zone.

    "resets 12:30pm (America/Chicago)" / "resets 5pm (UTC)": Claude's subscription session and
    weekly limits say when the window reopens as a local wall-clock time, which the delta
    grammars above cannot read — without this the cooldown arms a generic 60s, the primary is
    "restored" a minute later, and every turn re-burns a full-context 429. The zone is parsed
    via ``zoneinfo``; an unknown zone (or an impossible 12h time) returns None so the caller
    keeps its pre-existing behavior. A parsed instant already in the past is read as the NEXT
    day's window EXCEPT within a small tolerance — an error that surfaces seconds after the
    stated minute ("resets 12:30pm" read at 12:30:05) means the window has just reopened, and
    must not wait ~24h. The result is capped at a day.
    """
    try:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        meridiem = m.group(3).lower()
    except (TypeError, ValueError):
        return None
    if not 1 <= hour <= 12 or not 0 <= minute <= 59:
        return None
    hour = hour % 12 + (12 if meridiem == "pm" else 0)
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(m.group(4))
    except Exception:
        return None  # Unknown/invalid zone name: fail open to the delta grammars / default backoff.
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    delay = (target - now).total_seconds()
    if delay < 0:
        if -delay <= _RESET_CLOCK_JUST_PAST_TOLERANCE_S:
            delay = 0.0  # Just passed: the window is reopening now, not tomorrow.
        else:
            target += timedelta(days=1)
            delay = (target - now).total_seconds()
    return min(delay, _RESET_CLOCK_MAX_S)


# An explicit "retry after N s" wins over "resets in ..." (the credential pool's precedence):
# a body carrying both describes a short throttle inside a long quota window, and the
# shorter explicit wait is the one the provider actually asks for. The wall-clock grammar
# rides last: a delta and an absolute time in one body describe the same instant, and the
# delta is unambiguous about whose clock.
RETRY_DELAY_PATTERNS = (
    (_QUOTA_RESET_DELAY_RE, _quota_reset_seconds),
    (_RETRY_AFTER_SECONDS_RE, lambda m: float(m.group(1))),
    (_RESETS_IN_SECONDS_FIELD_RE, lambda m: float(m.group(1))),
    (_RESETS_IN_RE, _resets_in_seconds),
    (_RESETS_CLOCK_RE, _resets_clock_seconds),
)


def format_reset_window(seconds: float) -> str:
    """``~9h`` / ``~45 min`` for chat copy naming when a quota window reopens (ceilinged)."""
    seconds = int(seconds)
    return f"~{-(-seconds // 3600)}h" if seconds >= 3600 else f"~{-(-seconds // 60)} min"


def reset_delay_from_message(message: str) -> Optional[float]:
    """Seconds-until-reset parsed from free-text provider error messages, or None."""
    if not message:
        return None
    for pattern, to_seconds in RETRY_DELAY_PATTERNS:
        m = pattern.search(message)
        if m and (seconds := to_seconds(m)) is not None:
            return seconds
    return None


def jittered_backoff(attempt: int, *, base_delay: float = 5.0, max_delay: float = 120.0, jitter_ratio: float = 0.5) -> float:
    """min(base * 2^(attempt-1), max_delay) + uniform jitter in
    [0, jitter_ratio * delay]. ``attempt`` is 1-based."""
    global _jitter_counter
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    exponent = max(0, attempt - 1)
    delay = max_delay if (exponent >= 63 or base_delay <= 0) else min(base_delay * (2 ** exponent), max_delay)

    # Seed from time + counter so coarse clocks still decorrelate.
    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    return delay + random.Random(seed).uniform(0, jitter_ratio * delay)


def _error_text(error: Any) -> str:
    """Best-effort flattened provider error text for retry classification."""
    parts = [error, getattr(error, "message", None), getattr(error, "body", None), getattr(error, "response", None)]
    return " ".join(str(part) for part in parts if part is not None).lower()


def is_zai_coding_overload_error(*, base_url: str | None, model: str | None, error: Any) -> bool:
    """True only for the narrow Z.AI Coding Plan overload shape (429 + code
    1305 / "temporarily overloaded"), so ordinary quota 429s still fail fast."""
    text = _error_text(error)
    return (
        getattr(error, "status_code", None) == 429
        and "api.z.ai/api/coding/paas/v4" in (base_url or "").lower()
        and "glm-5.2" in (model or "").lower()
        and ("1305" in text or "temporarily overloaded" in text)
    )


def adaptive_rate_limit_backoff(
    attempt: int, *, base_url: str | None, model: str | None, error: Any, default_wait: float,
    short_attempts: int = _ZAI_CODING_OVERLOAD_SHORT_ATTEMPTS,
) -> tuple[float, str | None]:
    """``(wait_seconds, reason_label)``: ``default_wait`` for most providers; Z.AI Coding GLM-5.2 overloads keep
    ``short_attempts`` short retries, then 30→60→90→120s with light jitter. ``attempt`` is 1-based."""
    if not is_zai_coding_overload_error(base_url=base_url, model=model, error=error):
        return default_wait, None
    if attempt <= short_attempts:
        return default_wait, "zai_coding_overload_short"
    idx = min(attempt - short_attempts - 1, len(_ZAI_CODING_OVERLOAD_LONG_BACKOFF) - 1)
    base_delay = _ZAI_CODING_OVERLOAD_LONG_BACKOFF[idx]
    return jittered_backoff(1, base_delay=base_delay, max_delay=base_delay, jitter_ratio=0.2), "zai_coding_overload_long"


def zai_coding_overload_retry_ceiling(short_attempts: int = _ZAI_CODING_OVERLOAD_SHORT_ATTEMPTS) -> int:
    """Retry-loop ceiling for the full Z.AI overload schedule: one past the last long entry,
    because the loop gives up when ``retry_count >= ceiling`` BEFORE computing the attempt's
    backoff (the default ``api_max_retries`` of 3 equals ``short_attempts``)."""
    return short_attempts + len(_ZAI_CODING_OVERLOAD_LONG_BACKOFF) + 1
