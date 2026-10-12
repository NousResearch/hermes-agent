"""Retry utilities — jittered backoff for decorrelated retries.

Jittered delays (vs. fixed exponential) prevent thundering-herd retry spikes
when many sessions hit the same rate-limited provider concurrently.
"""

import random
import re
import threading
import time
from datetime import datetime, timezone, UTC
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

# Console Go relay (opencode.ai zen/go, shared by every Go model) overloads:
# the relay answers 503 / ``service_overloaded`` — and overload-signalled
# 429s — when saturated. Retrying those on a 2s hot loop burns quota against
# a window measured in minutes, so after ``_CONSOLE_GO_OVERLOAD_SHORT_ATTEMPTS``
# normal retries the wait parks the turn patiently instead. The short-count +
# jittered-ladder shape mirrors the Z.AI pattern above, but the ladder starts
# at minutes and the ceiling *reduces* total attempts (each wait is huge)
# instead of extending them.
_CONSOLE_GO_OVERLOAD_LONG_BACKOFF = (300.0, 1200.0, 3600.0)
_CONSOLE_GO_OVERLOAD_SHORT_ATTEMPTS = 1
_CONSOLE_GO_BASE_URL_TOKEN = "zen/go"
_CONSOLE_GO_OVERLOAD_TEXT_TOKENS = (
    "service_overloaded",
    "service overloaded",
    "service is overloaded",
    "temporarily overloaded",
    "server overloaded",
    "upstream overloaded",
)
# Park clamp: no structured-reset park ever sleeps longer than this, whatever
# the server claims. Interactive turns cap further (see turn_recovery).
CONSOLE_GO_PARK_MAX_S = 3600.0
# Epoch-millisecond threshold, kept identical to
# ``credential_pool._parse_absolute_timestamp`` so the two cannot desync.
_ABSOLUTE_TIMESTAMP_MS_THRESHOLD = 1_000_000_000_000


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
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


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


def _quota_reset_seconds(m: re.Match[str]) -> float:
    value = float(m.group(1))
    return value / 1000.0 if m.group(2).lower() == "ms" else value


def _resets_in_seconds(m: re.Match[str]) -> Optional[float]:
    if not any(m.groups()):  # "resets in" with no unit-bearing number: not this grammar
        return None
    return float(m.group(1) or 0) * 3600 + float(m.group(2) or 0) * 60 + float(m.group(3) or 0)


# An explicit "retry after N s" wins over "resets in ..." (the credential pool's precedence):
# a body carrying both describes a short throttle inside a long quota window, and the
# shorter explicit wait is the one the provider actually asks for.
RETRY_DELAY_PATTERNS = (
    (_QUOTA_RESET_DELAY_RE, _quota_reset_seconds),
    (_RETRY_AFTER_SECONDS_RE, lambda m: float(m.group(1))),
    (_RESETS_IN_SECONDS_FIELD_RE, lambda m: float(m.group(1))),
    (_RESETS_IN_RE, _resets_in_seconds),
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


# A wait longer than this is one a person feels: a non-rate-limit Retry-After this long is announced
# when it starts, and on the Nous free tier an attended session ends the turn instead of sitting through it.
LIVE_RETRY_WAIT_CAP_S = 60.0
# Anthropic Tier 1 input-token buckets reset in ~171s, so a 120s cap re-tripped the limit; 600s
# covers realistic provider windows while still rejecting pathological values (#26293).
RETRY_AFTER_CAP_S = 600.0


def provider_retry_after_seconds(error: Any) -> Optional[float]:
    """Provider-declared cooldown: the ``Retry-After`` header, else a ``retry_after`` body field
    (top level or nested under ``error``). None when absent, unparseable or zero: a zero or expired
    cooldown carries no usable wait, and treating it as one would hot-loop the provider."""
    value = parse_retry_after_seconds(getattr(getattr(error, "response", None), "headers", None))
    if value is None:
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            nested = body.get("error")
            value = parse_retry_after_seconds((nested if isinstance(nested, dict) else body).get("retry_after"))
    return value if value is not None and value > 0 else None


def is_console_go_overload_error(*, base_url: str | None, model: str | None, error: Any) -> bool:
    """True only for the narrow Console Go relay overload shape, so ordinary quota
    429s still fail fast (same narrowness contract as ``is_zai_coding_overload_error``).

    The relay serves every Go model under one base URL, so no model restriction:
    a 503 on the relay *is* the overload signal; a 429 qualifies only with
    overload text (``service_overloaded`` et al). ``model`` is accepted for
    call-signature parity and ignored."""
    _ = model
    if _CONSOLE_GO_BASE_URL_TOKEN not in (base_url or "").lower():
        return False
    if getattr(error, "status_code", None) == 503:
        return True
    if getattr(error, "status_code", None) != 429:
        return False
    text = _error_text(error)
    return any(token in text for token in _CONSOLE_GO_OVERLOAD_TEXT_TOKENS)


def console_go_overload_backoff(
    attempt: int, *, error: Any, default_wait: float,
    short_attempts: int = _CONSOLE_GO_OVERLOAD_SHORT_ATTEMPTS,
) -> tuple[float, str | None]:
    """``(wait_seconds, reason_label)``: ``default_wait`` for the first
    ``short_attempts`` attempts, then 300→1200→3600s with light jitter.
    ``attempt`` is 1-based. ``error`` is accepted for call-signature parity
    with ``adaptive_rate_limit_backoff`` and ignored."""
    _ = error
    if attempt <= short_attempts:
        return default_wait, "console_go_overload_short"
    idx = min(attempt - short_attempts - 1, len(_CONSOLE_GO_OVERLOAD_LONG_BACKOFF) - 1)
    base_delay = _CONSOLE_GO_OVERLOAD_LONG_BACKOFF[idx]
    return jittered_backoff(1, base_delay=base_delay, max_delay=base_delay, jitter_ratio=0.2), "console_go_overload_long"


def console_go_overload_retry_ceiling(short_attempts: int = _CONSOLE_GO_OVERLOAD_SHORT_ATTEMPTS) -> int:
    """Retry-loop ceiling for the full Console Go overload schedule: one past the last
    long entry, because the loop gives up when ``retry_count >= ceiling`` BEFORE computing
    the attempt's backoff. Apply with ``min()``, not ``max()`` — each wait is minutes
    long, so this ceiling *reduces* total attempts so the turn reaches fallback/surface
    instead of parking for hours (the inverse of the Z.AI ceiling, which extends)."""
    return short_attempts + len(_CONSOLE_GO_OVERLOAD_LONG_BACKOFF) + 1


def _error_payload_dicts(error: Any):
    """Yield the structured 429 body dicts: the top-level body, then a nested
    ``error`` object when present (the same unwrap ``extract_api_error_context``
    uses)."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        yield body
        nested = body.get("error")
        if isinstance(nested, dict):
            yield nested


def _seconds_until_timestamp(value: Any, *, now: float) -> Optional[float]:
    """Seconds from ``now`` until an absolute ``resets_at`` value: epoch seconds,
    epoch milliseconds (same ``> 1e12`` rule as the credential pool), or an
    ISO-8601 / HTTP-date string (naive datetimes read as UTC). None when
    unparseable. May be <= 0 when already expired — the caller treats that as
    absent so an expired reset never freezes the loop."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        epoch = float(value)
        if epoch > _ABSOLUTE_TIMESTAMP_MS_THRESHOLD:
            epoch /= 1000.0
        return epoch - now
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            when = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                when = parsedate_to_datetime(text)
            except (TypeError, ValueError):
                return None
        if when is None:  # older stdlib returns None instead of raising
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return when.timestamp() - now
    return None


def park_seconds_from_error(
    error: Any, *, now: Optional[float] = None, max_park_s: float = CONSOLE_GO_PARK_MAX_S,
) -> Optional[float]:
    """Seconds to park until a structured 429 reset: ``resets_in_seconds``
    (duration) or ``resets_at`` (absolute timestamp, s/ms/ISO) from the error
    body — the shortest credible positive candidate, clamped to ``max_park_s``.

    None when absent, unparseable, or already expired: the caller must fall
    through to the existing Retry-After path, never freeze."""
    now = time.time() if now is None else now
    candidates: list[float] = []
    for payload in _error_payload_dicts(error):
        raw_duration = payload.get("resets_in_seconds")
        if isinstance(raw_duration, (int, float)) and not isinstance(raw_duration, bool):
            candidates.append(float(raw_duration))
        until = _seconds_until_timestamp(payload.get("resets_at"), now=now)
        if until is not None:
            candidates.append(until)
    positive = [c for c in candidates if c > 0]
    if not positive:
        return None
    return min(min(positive), max_park_s)
