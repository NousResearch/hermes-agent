"""Retry utilities — jittered backoff for decorrelated retries.

Replaces fixed exponential backoff with jittered delays to prevent
thundering-herd retry spikes when multiple sessions hit the same
rate-limited provider concurrently.
"""

import random
import threading
import time

from agent.error_classifier import FailoverReason

# Monotonic counter for jitter seed uniqueness within the same process.
# Protected by a lock to avoid race conditions in concurrent retry paths
# (e.g. multiple gateway sessions retrying simultaneously).
_jitter_counter = 0
_jitter_lock = threading.Lock()


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


def select_retry_wait_time(
    attempt: int,
    *,
    reason: FailoverReason | str | None = None,
    retry_after: float | int | None = None,
) -> float:
    """Choose a retry delay tuned to the failure class.

    Overload errors (503/529) need a noticeably longer cool-down than generic
    transport failures; otherwise Hermes can burn through retries before the
    upstream cluster has time to recover.  Retry-After, when present, wins.
    """

    normalized_reason = getattr(reason, "value", reason) or ""

    if normalized_reason == FailoverReason.overloaded.value:
        max_delay = 180.0
        if retry_after is not None:
            try:
                return min(float(retry_after), max_delay)
            except (TypeError, ValueError):
                pass
        return jittered_backoff(attempt, base_delay=8.0, max_delay=max_delay)

    if normalized_reason in (FailoverReason.rate_limit.value, FailoverReason.billing.value):
        max_delay = 120.0
        if retry_after is not None:
            try:
                return min(float(retry_after), max_delay)
            except (TypeError, ValueError):
                pass
        return jittered_backoff(attempt, base_delay=2.0, max_delay=60.0)

    return jittered_backoff(attempt, base_delay=2.0, max_delay=60.0)
