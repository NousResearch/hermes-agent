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


def adaptive_rate_limit_backoff(
    attempt: int,
    *,
    base_url: str | None,
    model: str | None,
    error: Any,
    default_wait: float,
    short_attempts: int = 3,
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
    if not is_zai_coding_overload_error(base_url=base_url, model=model, error=error):
        return default_wait, None
    if attempt <= short_attempts:
        return default_wait, "zai_coding_overload_short"

    idx = min(attempt - short_attempts - 1, len(_ZAI_CODING_OVERLOAD_LONG_BACKOFF) - 1)
    base_delay = _ZAI_CODING_OVERLOAD_LONG_BACKOFF[idx]
    # A smaller jitter ratio keeps long waits readable while still avoiding
    # synchronized retry storms across concurrent Hermes sessions.
    return jittered_backoff(1, base_delay=base_delay, max_delay=base_delay, jitter_ratio=0.2), "zai_coding_overload_long"


# Cap for a honored Retry-After, by class. A rate-limit reset window can be
# minutes (Anthropic Tier-1 input buckets reset ~171s, some providers longer),
# so 600s. A provider OVERLOAD / local-relay backpressure transient clears in
# seconds, so a much tighter 60s cap is sufficient and safer.
RETRY_AFTER_CAP_RATE_LIMIT_S = 600.0
RETRY_AFTER_CAP_OVERLOAD_S = 60.0


def resolve_retry_after(
    *,
    raw_value: Any,
    is_rate_limit: bool,
    is_overload: bool,
    retry_count: int,
    max_retries: int,
) -> float | None:
    """Decide whether to honor a server ``Retry-After`` and for how long.

    Pure function (no I/O) so the honor-policy is unit-testable in isolation
    from the conversation loop. Returns the number of seconds to wait, or
    ``None`` to fall through to the caller's jittered backoff.

    Policy:
      * Honor only for a rate-limit OR a provider overload (503/529). A
        pool-at-capacity 503 from the local relay is a self-describing
        transient that emits a bounded ``Retry-After``; honoring it beats
        blind jitter. Any other reason → ``None`` (jitter).
      * The caller activates its fallback chain at ``retry_count >=
        max_retries``, so the honor-reachable retries are ``1 .. max_retries-1``.
        Reserve the LAST reachable retry for jitter (so the fast direct-box
        fallback isn't delayed by a relay-informed wait) — but ONLY when there
        are at least TWO reachable retries to spare one. When ``max_retries==2``
        there is a single reachable retry; skipping it would make the whole
        overload feature a no-op, so it IS honored (Greptile #223 P1).
      * A non-numeric value (e.g. an HTTP-date ``Retry-After``, RFC-valid but
        not a bare number) is NOT parsed here → ``None`` (jitter). This is
        deliberate: overload/backpressure sources emit numeric seconds; date
        parsing would be scope creep.
      * The honored value is clamped to a class-specific cap
        (rate-limit 600s, overload 60s) and floored at 0.

    ``retry_count`` is the caller's 1-based attempt number (incremented before
    this runs); ``max_retries`` is the retry ceiling.
    """
    if not (is_rate_limit or is_overload):
        return None
    # Past/at the fallback threshold → don't honor (defensive; the caller
    # normally activates fallback before reaching here).
    if retry_count >= max_retries:
        return None
    # Reserve the final reachable retry for a fast jitter→fallback, but only
    # when there are ≥2 reachable retries (max_retries ≥ 3) so we don't spend
    # our only reachable retry and disable the feature (Greptile #223 P1).
    if max_retries >= 3 and retry_count >= max_retries - 1:
        return None
    if raw_value in (None, ""):
        return None
    try:
        secs = float(raw_value)
    except (TypeError, ValueError):
        return None
    if secs <= 0:
        return None
    cap = RETRY_AFTER_CAP_RATE_LIMIT_S if is_rate_limit else RETRY_AFTER_CAP_OVERLOAD_S
    return min(secs, cap)
