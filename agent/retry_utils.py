"""Retry utilities — jittered backoff for decorrelated retries.

Replaces fixed exponential backoff with jittered delays to prevent
thundering-herd retry spikes when multiple sessions hit the same
rate-limited provider concurrently.
"""

import random
import threading
import time
from typing import Any

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
_NVIDIA_NIM_RATE_LIMIT_BACKOFF = (15.0, 30.0, 60.0, 120.0)

# Number of initial short retries before the adaptive long-backoff tier kicks
# in. Shared by ``adaptive_rate_limit_backoff`` (which walks the long table
# starting at attempt ``short_attempts + 1``) and
# ``zai_coding_overload_retry_ceiling`` (which sizes the retry loop so every
# long-tier entry is reachable). Keeping it a single module constant prevents
# the two from silently desyncing if the short-retry count is ever tuned.
_ZAI_CODING_OVERLOAD_SHORT_ATTEMPTS = 3


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


def is_nvidia_nim_rate_limit_error(*, base_url: str | None, error: Any) -> bool:
    """Return True for NVIDIA NIM endpoint 429s.

    NVIDIA NIM can return a terse ``{"status":429,"title":"Too Many Requests"}``
    body without a Retry-After header. Retrying after the generic 2s backoff
    tends to create retry storms across concurrent workers, so use a wider
    provider-specific window while keeping ordinary non-NVIDIA 429s unchanged.
    """
    base = (base_url or "").rstrip("/").lower()
    status = getattr(error, "status_code", None)
    text = _error_text(error)
    return (
        "integrate.api.nvidia.com/v1" in base
        and (
            status == 429
            or "too many requests" in text
            or '"status":429' in text
            or "'status': 429" in text
        )
    )


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
    waits (30s → 60s → 90s → 120s, capped) plus light jitter.

    ``attempt`` is 1-based, matching the retry loop's logged attempt number.
    Returns ``(wait_seconds, reason_label)`` where ``reason_label`` is suitable
    for status/log decoration when a provider-specific policy fired.
    """
    if is_zai_coding_overload_error(base_url=base_url, model=model, error=error):
        if attempt <= short_attempts:
            return default_wait, "zai_coding_overload_short"

        idx = min(attempt - short_attempts - 1, len(_ZAI_CODING_OVERLOAD_LONG_BACKOFF) - 1)
        base_delay = _ZAI_CODING_OVERLOAD_LONG_BACKOFF[idx]
        # A smaller jitter ratio keeps long waits readable while still avoiding
        # synchronized retry storms across concurrent Hermes sessions.
        return jittered_backoff(1, base_delay=base_delay, max_delay=base_delay, jitter_ratio=0.2), "zai_coding_overload_long"

    if is_nvidia_nim_rate_limit_error(base_url=base_url, error=error):
        idx = min(max(attempt, 1) - 1, len(_NVIDIA_NIM_RATE_LIMIT_BACKOFF) - 1)
        base_delay = _NVIDIA_NIM_RATE_LIMIT_BACKOFF[idx]
        return jittered_backoff(1, base_delay=base_delay, max_delay=base_delay, jitter_ratio=0.2), "nvidia_nim_rate_limit"

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
