"""Retry utilities — jittered backoff for decorrelated retries.

Replaces fixed exponential backoff with jittered delays to prevent
thundering-herd retry spikes when multiple sessions hit the same
rate-limited provider concurrently. Also provides proactive rate limit
awareness via x-ratelimit-* headers.
"""

import random
import threading
import time
from typing import Any, Mapping, Optional

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


def extract_rate_limit_reset_seconds(
    headers: Mapping[str, str],
    *,
    retry_after: Optional[float] = None,
    max_cap: float = 300.0,
    provider: str = "openrouter",
) -> Optional[float]:
    """Extract and prioritize rate limit reset time from headers.

    Prefers x-ratelimit-reset-requests (OpenRouter/OpenAI-compatible)
    over generic Retry-After, as it's more precise. Both are capped
    at max_cap to prevent indefinite waits during daily limits.

    Args:
        headers: Response headers (case-insensitive).
        retry_after: Pre-parsed Retry-After value in seconds (fallback).
        max_cap: Maximum wait time in seconds (default 300s = 5 min).
        provider: Provider name for logging context.

    Returns:
        Wait time in seconds, or None if no rate limit info found.
    """
    if not headers:
        return retry_after

    # Normalize headers to lowercase for case-insensitive lookup
    lowered = {k.lower(): v for k, v in headers.items()}

    # Prefer x-ratelimit-reset-requests (RPM window reset time)
    # or x-ratelimit-reset-requests-1h (hourly window reset time).
    # OpenRouter populates both; we use the minute window first as
    # it's typically the tighter constraint for free tier (20 RPM).
    for header_name in [
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-requests-1h",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset-tokens-1h",
    ]:
        raw_value = lowered.get(header_name)
        if raw_value:
            try:
                reset_seconds = float(raw_value)
                if reset_seconds > 0:
                    # Cap to prevent infinitely long waits
                    capped = min(reset_seconds, max_cap)
                    return capped
            except (TypeError, ValueError):
                continue

    # Fallback to Retry-After if provided
    if retry_after is not None and retry_after > 0:
        return min(retry_after, max_cap)

    return None


def check_rate_limit_headroom(
    headers: Mapping[str, str],
    *,
    threshold: int = 2,
) -> Optional[float]:
    """Check if rate limit headroom is critically low.

    Returns the reset time in seconds if remaining requests/tokens
    fall below threshold, indicating proactive throttling is needed.
    Used to avoid unnecessary 429 responses.

    Args:
        headers: Response headers from previous request.
        threshold: Alert if remaining requests ≤ this value (default 2).

    Returns:
        Reset time in seconds if low headroom, None otherwise.
    """
    if not headers:
        return None

    lowered = {k.lower(): v for k, v in headers.items()}

    # Check requests/min headroom first (tighter for free tier)
    try:
        remaining_req = int(float(lowered.get("x-ratelimit-remaining-requests", threshold + 1)))
        if remaining_req <= threshold:
            reset = lowered.get("x-ratelimit-reset-requests")
            if reset:
                return float(reset)
    except (TypeError, ValueError):
        pass

    # Check tokens/min as secondary signal
    try:
        remaining_tok = int(float(lowered.get("x-ratelimit-remaining-tokens", threshold + 1)))
        if remaining_tok <= threshold:
            reset = lowered.get("x-ratelimit-reset-tokens")
            if reset:
                return float(reset)
    except (TypeError, ValueError):
        pass

    return None
