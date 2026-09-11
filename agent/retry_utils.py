"""Retry utilities — jittered backoff for decorrelated retries.

Jittered delays (vs. fixed exponential) prevent thundering-herd retry spikes
when many sessions hit the same rate-limited provider concurrently.
"""

import random
import threading
import time
from datetime import datetime, timezone
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

# Anthropic surfaces capacity dips as an `overloaded_error` delivered *inside*
# a streaming HTTP 200 body — no 429/529 status, no Retry-After header. Field
# logs show whole 2-5 minute windows in which every request fails regardless of
# context size (a fresh 9K-token session fails alongside a 168K one), then the
# window clears on its own. The default 3-retry budget (~25s of 2s-base
# exponential backoff) always expires inside such a window, so the turn dies
# with "API call failed after 3 retries: HTTP 200: Overloaded" even though a
# slightly wider wait would have succeeded. Same remedy as the Z.AI overload
# above: a few short retries, then a widening long tier.
_ANTHROPIC_OVERLOAD_LONG_BACKOFF = (15.0, 30.0, 60.0, 120.0)
_ANTHROPIC_OVERLOAD_SHORT_ATTEMPTS = 3


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


def is_anthropic_overload_error(*, base_url: str | None, model: str | None, error: Any) -> bool:
    """Return True for Anthropic-wire transient capacity overloads.

    Anthropic reports these as ``{"type": "overloaded_error"}``. On the
    streaming path the envelope is an HTTP 200 whose body carries the error, so
    matching on status alone misses them entirely — which is precisely why the
    generic 503/529 handling never fires for them.

    Deliberately *not* scoped to ``api.anthropic.com``. Anthropic-compatible
    endpoints reached through ``ANTHROPIC_BASE_URL`` — proxies, gateways, and
    third-party models served over the Anthropic wire format — emit the same
    error type and dead-end for the same reason; the endpoint in #55540 is
    exactly such a proxy. Scoping by host would reintroduce the same class of
    blind spot this function exists to close. ``overloaded_error`` is a
    distinctive Anthropic-wire token, so matching the body rather than the host
    does not sweep in other providers' free-text wording, and Z.AI's own
    429/1305 overload is matched earlier in ``adaptive_rate_limit_backoff`` and
    keeps its dedicated schedule.

    ``base_url`` and ``model`` are accepted for call-site symmetry with
    ``is_zai_coding_overload_error`` and are intentionally unused.
    """
    return "overloaded_error" in _error_text(error)


def _overload_long_backoff(
    attempt: int,
    *,
    short_attempts: int,
    table: tuple[float, ...],
    label: str,
    default_wait: float,
) -> tuple[float, str]:
    """Short retries first, then walk ``table`` one entry per later attempt."""
    if attempt <= short_attempts:
        return default_wait, f"{label}_short"

    idx = min(attempt - short_attempts - 1, len(table) - 1)
    base_delay = table[idx]
    # A smaller jitter ratio keeps long waits readable while still avoiding
    # synchronized retry storms across concurrent Hermes sessions.
    wait = jittered_backoff(1, base_delay=base_delay, max_delay=base_delay, jitter_ratio=0.2)
    return wait, f"{label}_long"


def adaptive_rate_limit_backoff(
    attempt: int, *, base_url: str | None, model: str | None, error: Any, default_wait: float,
    short_attempts: int = _ZAI_CODING_OVERLOAD_SHORT_ATTEMPTS,
) -> tuple[float, str | None]:
    """``(wait_seconds, reason_label)``: ``default_wait`` for most providers; Z.AI Coding GLM-5.2 overloads keep
    ``short_attempts`` short retries, then 30→60→90→120s with light jitter; Anthropic
    ``overloaded_error`` (streaming HTTP 200 envelope) gets the same treatment on 15/30/60/120s.
    ``attempt`` is 1-based."""
    if is_zai_coding_overload_error(base_url=base_url, model=model, error=error):
        return _overload_long_backoff(
            attempt,
            short_attempts=short_attempts,
            table=_ZAI_CODING_OVERLOAD_LONG_BACKOFF,
            label="zai_coding_overload",
            default_wait=default_wait,
        )
    if is_anthropic_overload_error(base_url=base_url, model=model, error=error):
        return _overload_long_backoff(
            attempt,
            short_attempts=_ANTHROPIC_OVERLOAD_SHORT_ATTEMPTS,
            table=_ANTHROPIC_OVERLOAD_LONG_BACKOFF,
            label="anthropic_overload",
            default_wait=default_wait,
        )
    return default_wait, None


def zai_coding_overload_retry_ceiling(short_attempts: int = _ZAI_CODING_OVERLOAD_SHORT_ATTEMPTS) -> int:
    """Retry-loop ceiling for the full Z.AI overload schedule: one past the last long entry,
    because the loop gives up when ``retry_count >= ceiling`` BEFORE computing the attempt's
    backoff (the default ``api_max_retries`` of 3 equals ``short_attempts``)."""
    return short_attempts + len(_ZAI_CODING_OVERLOAD_LONG_BACKOFF) + 1


def anthropic_overload_retry_ceiling(short_attempts: int = _ANTHROPIC_OVERLOAD_SHORT_ATTEMPTS) -> int:
    """Retry-loop ceiling needed for the full Anthropic overload schedule.

    Same rationale as ``zai_coding_overload_retry_ceiling``: the loop gives up
    once ``retry_count >= ceiling``, and that check runs before the attempt's
    backoff is computed, so the ceiling must sit one past the final long-backoff
    entry. With the default ``api_max_retries`` (3) the long tier is otherwise
    unreachable and a capacity window that lasts minutes kills the turn.
    """
    return short_attempts + len(_ANTHROPIC_OVERLOAD_LONG_BACKOFF) + 1
