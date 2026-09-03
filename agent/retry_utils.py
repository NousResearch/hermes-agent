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

# Some providers report transient *capacity* pressure as a 429 that a short
# retry cannot clear: Z.AI Coding Plan's GLM-5.2 endpoint returns code 1305
# ("The service may be temporarily overloaded..."), and Cloudflare Workers AI
# returns code 3040 ("Capacity temporarily exceeded, please try again") under
# account contention. Short retries just hammer the same window; after a few
# normal retries, progressively widen the wait. Keep the cap
# interactive-friendly: a simple TUI message should fail visibly in minutes,
# not sit silent for 20+ minutes.
_CAPACITY_OVERLOAD_LONG_BACKOFF = (30.0, 60.0, 90.0, 120.0)

# Number of initial short retries before the adaptive long-backoff tier kicks
# in. Shared by ``adaptive_rate_limit_backoff`` (which walks the long table
# starting at attempt ``short_attempts + 1``) and
# ``capacity_overload_retry_ceiling`` (which sizes the retry loop so every
# long-tier entry is reachable). Keeping it a single module constant prevents
# the two from silently desyncing if the short-retry count is ever tuned.
_CAPACITY_OVERLOAD_SHORT_ATTEMPTS = 3

# Back-compat aliases for the Z.AI-only names these constants shipped under.
_ZAI_CODING_OVERLOAD_LONG_BACKOFF = _CAPACITY_OVERLOAD_LONG_BACKOFF
_ZAI_CODING_OVERLOAD_SHORT_ATTEMPTS = _CAPACITY_OVERLOAD_SHORT_ATTEMPTS


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


def is_cloudflare_capacity_error(*, base_url: str | None, error: Any) -> bool:
    """Return True for Cloudflare Workers AI transient capacity 429s.

    Workers AI reports account-level contention as HTTP 429 with body code
    3040 and message "Capacity temporarily exceeded, please try again"
    (surfaced as ``AiError``). Like the Z.AI overload shape, retrying against
    the same short window rarely helps — the account needs time to drain.

    The match is deliberately narrow (Cloudflare host + that one message/code)
    so ordinary quota/billing 429s keep failing fast through the existing
    classifier. Unlike the Z.AI check there is no model constraint: every
    ``@cf/...`` model on the account shares the same capacity pool.
    """
    base = (base_url or "").lower()
    status = getattr(error, "status_code", None)
    text = _error_text(error)
    return (
        status == 429
        and "api.cloudflare.com" in base
        and ("3040" in text or "capacity temporarily exceeded" in text)
    )


def capacity_overload_policy(*, base_url: str | None, model: str | None, error: Any) -> str | None:
    """Name the provider capacity-overload policy an error qualifies for.

    Returns ``"zai_coding_overload"``, ``"cloudflare_capacity"``, or ``None``
    when no provider-specific long-backoff policy applies. Callers use this to
    gate both the adaptive backoff and the extended retry ceiling.
    """
    if is_zai_coding_overload_error(base_url=base_url, model=model, error=error):
        return "zai_coding_overload"
    if is_cloudflare_capacity_error(base_url=base_url, error=error):
        return "cloudflare_capacity"
    return None


def adaptive_rate_limit_backoff(
    attempt: int,
    *,
    base_url: str | None,
    model: str | None,
    error: Any,
    default_wait: float,
    short_attempts: int = _CAPACITY_OVERLOAD_SHORT_ATTEMPTS,
) -> tuple[float, str | None]:
    """Provider-aware rate-limit backoff.

    For most providers this returns ``default_wait`` unchanged. For provider
    capacity overloads (Z.AI Coding Plan GLM-5.2 1305s, Cloudflare Workers AI
    3040s), keep the first ``short_attempts`` retries on the normal short
    exponential schedule, then switch to progressively longer waits
    (30s → 60s → 90s → 120s, capped) plus light jitter.

    ``attempt`` is 1-based, matching the retry loop's logged attempt number.
    Returns ``(wait_seconds, reason_label)`` where ``reason_label`` is
    ``"<policy>_short"`` / ``"<policy>_long"`` and is suitable for status/log
    decoration when a provider-specific policy fired.
    """
    policy = capacity_overload_policy(base_url=base_url, model=model, error=error)
    if policy is None:
        return default_wait, None
    if attempt <= short_attempts:
        return default_wait, f"{policy}_short"

    idx = min(attempt - short_attempts - 1, len(_CAPACITY_OVERLOAD_LONG_BACKOFF) - 1)
    base_delay = _CAPACITY_OVERLOAD_LONG_BACKOFF[idx]
    # A smaller jitter ratio keeps long waits readable while still avoiding
    # synchronized retry storms across concurrent Hermes sessions.
    return (
        jittered_backoff(1, base_delay=base_delay, max_delay=base_delay, jitter_ratio=0.2),
        f"{policy}_long",
    )


def capacity_overload_retry_ceiling(short_attempts: int = _CAPACITY_OVERLOAD_SHORT_ATTEMPTS) -> int:
    """Retry-loop ceiling needed for the full capacity-overload backoff schedule.

    The adaptive policy runs ``short_attempts`` short retries, then walks the
    long-backoff table one entry per subsequent attempt. The retry loop gives
    up as soon as ``retry_count >= ceiling`` — and that check runs *before* the
    attempt's backoff is computed — so the ceiling must sit one past the final
    long-backoff entry for every long tier to actually execute.

    With the default ``agent.api_max_retries`` (3) equal to ``short_attempts``
    (3), the loop always gave up before reaching the long tier, leaving the
    whole long-backoff schedule as dead code. Callers extend the ceiling to
    this value for capacity-overload 429s (Z.AI 1305, Cloudflare 3040) so the
    30/60/90/120s waits run.
    """
    return short_attempts + len(_CAPACITY_OVERLOAD_LONG_BACKOFF) + 1


# Back-compat alias for the Z.AI-only name this helper shipped under. The
# ceiling is schedule-derived, so both policies share one value.
zai_coding_overload_retry_ceiling = capacity_overload_retry_ceiling
