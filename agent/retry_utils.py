"""Retry utilities — jittered backoff for decorrelated retries.

Replaces fixed exponential backoff with jittered delays to prevent
thundering-herd retry spikes when multiple sessions hit the same
rate-limited provider concurrently.
"""

import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

# Monotonic counter for jitter seed uniqueness within the same process.
# Protected by a lock to avoid race conditions in concurrent retry paths
# (e.g. multiple gateway sessions retrying simultaneously).
_jitter_counter = 0
_jitter_lock = threading.Lock()

# Z.AI Coding Plan's GLM-5.2 endpoint often returns HTTP 429 code 1305
# ("The service may be temporarily overloaded...") for otherwise valid
# Hermes requests. Short retries tend to hammer the same overloaded window;
# after a few normal retries, progressively widen the wait window. Keep the
# cap interactive-friendly: a simple TUI message should fail visibly in minutes,
# not sit silent for 20+ minutes.
_ZAI_CODING_OVERLOAD_LONG_BACKOFF = (30.0, 60.0, 90.0, 120.0)

# Number of initial short retries before the adaptive long-backoff tier kicks
# in. Shared by ``adaptive_rate_limit_backoff`` (which walks the long table
# starting at attempt ``short_attempts + 1``) and
# ``zai_coding_overload_retry_ceiling`` (which sizes the retry loop so every
# long-tier entry is reachable). Keeping it a single module constant prevents
# the two from silently desyncing if the short-retry count is ever tuned.
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
    """Parse a ``Retry-After`` value into non-negative seconds.

    Accepts either a raw header value (numeric string / HTTP-date / number)
    or a headers mapping, in which case the ``Retry-After`` key is looked up
    case-insensitively (``.get`` on dict-like objects tries both common
    casings; real HTTP header containers like httpx/requests are already
    case-insensitive).

    Returns:
        Seconds as a ``float`` (negative deltas clamped to ``0.0``), or
        ``None`` when the header is absent or unparseable.
    """
    raw = value_or_headers
    if raw is not None and not isinstance(raw, (str, int, float)):
        # Looks like a headers mapping — pull the header out of it.
        getter = getattr(raw, "get", None)
        if callable(getter):
            try:
                value = getter("Retry-After")
                if value is None:
                    value = getter("retry-after")
            except Exception:
                return None
            raw = value
        else:
            return None
    if raw is None:
        return None
    if isinstance(raw, bool):
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
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    jitter_ratio: float = 0.5,
) -> float:
    """Compute a jittered exponential backoff delay.

    Args:
        attempt: 1-based retry attempt number.
        base_delay: Base delay in seconds for attempt 1.
        max_delay: Maximum delay cap in seconds.
        jitter_ratio: Fraction of computed delay to use as random jitter
            range.  0.5 means jitter is uniform in [0, 0.5 * delay].

    Returns:
        Delay in seconds: min(base * 2^(attempt-1), max_delay) + jitter.

    The jitter decorrelates concurrent retries so multiple sessions
    hitting the same provider don't all retry at the same instant.
    """
    global _jitter_counter
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        delay = max_delay
    else:
        delay = min(base_delay * (2 ** exponent), max_delay)

    # Seed from time + counter for decorrelation even with coarse clocks.
    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    rng = random.Random(seed)
    jitter = rng.uniform(0, jitter_ratio * delay)

    return delay + jitter


def _error_text(error: Any) -> str:
    """Best-effort flattened provider error text for retry classification."""
    parts = [
        error,
        getattr(error, "message", None),
        getattr(error, "body", None),
        getattr(error, "response", None),
    ]
    return " ".join(str(part) for part in parts if part is not None).lower()


def is_zai_coding_overload_error(*, base_url: str | None, model: str | None, error: Any) -> bool:
    """Return True for Z.AI Coding Plan transient overload 429s.

    The coding-plan endpoint reports overload as HTTP 429 with body code 1305
    and message "The service may be temporarily overloaded...". Treat only
    that narrow shape specially so ordinary quota/billing 429s still fail fast
    through the existing classifier.
    """
    base = (base_url or "").lower()
    model_name = (model or "").lower()
    status = getattr(error, "status_code", None)
    text = _error_text(error)
    return (
        status == 429
        and "api.z.ai/api/coding/paas/v4" in base
        and "glm-5.2" in model_name
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
    attempt: int,
    *,
    base_url: str | None,
    model: str | None,
    error: Any,
    default_wait: float,
    short_attempts: int = _ZAI_CODING_OVERLOAD_SHORT_ATTEMPTS,
) -> tuple[float, str | None]:
    """Provider-aware rate-limit backoff.

    For most providers this returns ``default_wait`` unchanged. For Z.AI
    Coding Plan GLM-5.2 overloads, keep the first ``short_attempts`` retries on
    the normal short exponential schedule, then switch to progressively longer
    waits (30s → 60s → 90s → 120s, capped) plus light jitter. Anthropic
    ``overloaded_error`` responses get the same treatment on a 15/30/60/120s
    table.

    ``attempt`` is 1-based, matching the retry loop's logged attempt number.
    Returns ``(wait_seconds, reason_label)`` where ``reason_label`` is suitable
    for status/log decoration when a provider-specific policy fired.
    """
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
    """Retry-loop ceiling needed for the full Z.AI overload backoff schedule.

    The adaptive policy runs ``short_attempts`` short retries, then walks the
    long-backoff table one entry per subsequent attempt. The retry loop gives
    up as soon as ``retry_count >= ceiling`` — and that check runs *before* the
    attempt's backoff is computed — so the ceiling must sit one past the final
    long-backoff entry for every long tier to actually execute.

    With the default ``api_max_retries`` (3) equal to ``short_attempts`` (3),
    the loop always gave up before reaching the long tier, leaving the whole
    long-backoff schedule as dead code. Callers extend the ceiling to this
    value for Z.AI Coding overload 429s so the 30/60/90/120s waits run.
    """
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
