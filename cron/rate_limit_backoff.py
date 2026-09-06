"""Deterministic, durable backoff policy for provider rate-limit failures."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

_RATE_LIMIT_RE = re.compile(r"\b429\b|rate[ -]?limit|usage limit", re.IGNORECASE)
_RESET_AT_RE = re.compile(
    r"(?:limit\s+will\s+reset|reset[_ -]?at|resets?)\s*(?:at|[:=])?\s*"
    r"(?P<value>[^\n\r;}]+)",
    re.IGNORECASE,
)
_RETRY_AFTER_RE = re.compile(
    r"retry[- ]after\s*(?:[:=]|is)?\s*(?P<value>[^\n\r;}]+)",
    re.IGNORECASE,
)
_EPOCH_RESET_RE = re.compile(
    r"(?:x[- ]?rate[- ]?limit[- ]?reset|ratelimit[-_ ]reset)\s*[:=]\s*(?P<epoch>\d{10}(?:\.\d+)?)",
    re.IGNORECASE,
)

_EXPLICIT_JITTER_SECONDS = 300
_FALLBACK_JITTER_SECONDS = 60
_FALLBACK_BASE_SECONDS = 15 * 60
_FALLBACK_MAX_SECONDS = 6 * 60 * 60
_MAX_DECLARED_DELAY_SECONDS = 31 * 24 * 60 * 60


def _aware(value: datetime, now: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=now.tzinfo)
    return value


def _parse_datetime(value: str, now: datetime) -> Optional[datetime]:
    text = value.strip().strip("'\"")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return _aware(datetime.fromisoformat(text), now)
    except ValueError:
        pass
    try:
        return _aware(parsedate_to_datetime(text), now)
    except (TypeError, ValueError, OverflowError):
        pass
    # Some providers render quota resets for humans rather than as ISO/RFC
    # timestamps. Naive values inherit the provider error's local process zone,
    # matching the ISO-without-offset behavior above.
    normalized = re.sub(r"\s+at\s+", " ", text, flags=re.IGNORECASE)
    for fmt in ("%B %d, %Y %I:%M %p", "%b %d, %Y %I:%M %p"):
        try:
            return _aware(datetime.strptime(normalized, fmt), now)
        except ValueError:
            continue
    return None


def _declared_reset(error: str, now: datetime) -> Optional[datetime]:
    retry_after = _RETRY_AFTER_RE.search(error)
    if retry_after:
        value = retry_after.group("value").strip()
        try:
            candidate = now + timedelta(seconds=float(value))
        except (TypeError, ValueError, OverflowError):
            candidate = _parse_datetime(value, now)
        if candidate is not None:
            return candidate

    epoch_reset = _EPOCH_RESET_RE.search(error)
    if epoch_reset:
        try:
            candidate = datetime.fromtimestamp(
                float(epoch_reset.group("epoch")), tz=now.tzinfo
            )
        except (TypeError, ValueError, OverflowError, OSError):
            candidate = None
        if candidate is not None:
            return candidate

    match = _RESET_AT_RE.search(error)
    if match:
        return _parse_datetime(match.group("value"), now)
    return None


def _jitter_seconds(job_id: str, key: str, ceiling: int) -> int:
    digest = hashlib.sha256(f"{job_id}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % ceiling + 1


def plan_provider_backoff(
    job: Dict[str, Any], error: Optional[str], *, now: datetime
) -> Optional[Dict[str, Any]]:
    """Plan a no-inference retry delay for a provider rate-limit failure."""
    text = str(error or "")
    if job.get("no_agent") or not _RATE_LIMIT_RE.search(text):
        return None

    job_id = str(job.get("id") or "unknown")
    existing = job.get("provider_backoff")
    attempt = 1
    if isinstance(existing, dict):
        try:
            attempt = max(1, int(existing.get("attempt") or 0) + 1)
        except (TypeError, ValueError):
            attempt = 1

    reset_at = _declared_reset(text, now)
    max_reset = now + timedelta(seconds=_MAX_DECLARED_DELAY_SECONDS)
    if reset_at is not None and now < reset_at <= max_reset:
        key = reset_at.isoformat()
        until = reset_at + timedelta(
            seconds=_jitter_seconds(job_id, key, _EXPLICIT_JITTER_SECONDS)
        )
        return {
            "until": until.isoformat(),
            "reset_at": reset_at.isoformat(),
            "detected_at": now.isoformat(),
            "source": "provider_reset",
            "attempt": attempt,
        }

    delay = min(
        _FALLBACK_BASE_SECONDS * (2 ** min(attempt - 1, 16)),
        _FALLBACK_MAX_SECONDS,
    )
    jitter = _jitter_seconds(job_id, str(attempt), _FALLBACK_JITTER_SECONDS)
    return {
        "until": (now + timedelta(seconds=delay + jitter)).isoformat(),
        "reset_at": None,
        "detected_at": now.isoformat(),
        "source": "fallback",
        "attempt": attempt,
    }


def provider_backoff_active(job: Dict[str, Any], *, now: datetime) -> bool:
    """Return whether a persisted provider backoff still suppresses auto-fire."""
    backoff = job.get("provider_backoff")
    if not isinstance(backoff, dict):
        return False
    until = backoff.get("until")
    if not isinstance(until, str):
        return False
    try:
        parsed = _aware(datetime.fromisoformat(until), now)
        delay = (parsed - now).total_seconds()
    except (TypeError, ValueError):
        return False
    max_delay = _MAX_DECLARED_DELAY_SECONDS + _EXPLICIT_JITTER_SECONDS
    return 0 < delay <= max_delay
