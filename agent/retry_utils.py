"""Retry utilities — jittered backoff, retryable error classification.

Inspired by claw-code's Rust retry architecture (splitmix64 jitter,
centralized is_retryable_status, structured failure classification).
"""

import logging
import random
import re
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# HTTP status codes that warrant a retry attempt.
# Transient server errors + rate limits.  Client errors (4xx except 429/408/413)
# are intentionally excluded — they indicate a problem with the request itself.
RETRYABLE_STATUS_CODES = frozenset({408, 413, 429, 500, 502, 503, 504, 529})

# Error message phrases that indicate a transient transport failure
# (connection drop, timeout, pool exhaustion).  These are worth retrying
# with a fresh connection even when no HTTP status code is available.
TRANSIENT_TRANSPORT_PHRASES = frozenset({
    "readtimeout", "connecttimeout", "pooltimeout",
    "connecterror", "remoteprotocolerror",
    "connection lost", "connection reset", "connection closed",
    "connection terminated", "network error", "network connection",
    "peer closed", "broken pipe", "upstream connect error",
})

_RETRY_AFTER_TEXT_RE = re.compile(
    r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
    re.IGNORECASE,
)

# Monotonic counter for jitter seed uniqueness within the same process.
# Protected by a lock to avoid race conditions in concurrent retry paths
# (e.g. multiple sessions retrying simultaneously via the gateway).
_jitter_counter = 0
_jitter_lock = threading.Lock()


def _transient_exception_types() -> tuple[type[BaseException], ...]:
    """Best-effort import of common transient transport exception classes."""
    exc_types: list[type[BaseException]] = []

    try:
        import httpx

        exc_types.extend(
            [
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.PoolTimeout,
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                httpx.WriteError,
            ]
        )
    except Exception:
        pass

    try:
        import requests

        exc_types.extend(
            [
                requests.exceptions.ConnectionError,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ]
        )
    except Exception:
        pass

    # Deduplicate while preserving deterministic order.
    return tuple(dict.fromkeys(exc_types))


_TRANSIENT_EXCEPTION_TYPES = _transient_exception_types()


def is_retryable_status(status_code: Optional[int]) -> bool:
    """Return True if the HTTP status code indicates a retryable failure.

    Centralized classification — use this instead of scattered status code
    checks to keep retry policy consistent across the codebase.
    """
    if status_code is None:
        return False
    return status_code in RETRYABLE_STATUS_CODES


def is_transient_transport_error(error: Exception) -> bool:
    """Return True if the exception looks like a transient network failure.

    These are worth retrying with a fresh connection (rebuild client, clear
    connection pool) rather than backing off on the same dead connection.
    """
    if _TRANSIENT_EXCEPTION_TYPES and isinstance(error, _TRANSIENT_EXCEPTION_TYPES):
        return True

    # Prefer the first argument as the message payload; str(error) can include
    # wrapper/context text in some exception implementations.
    if error.args:
        error_str = str(error.args[0]).lower()
    else:
        error_str = str(error).lower()

    return any(phrase in error_str for phrase in TRANSIENT_TRANSPORT_PHRASES)


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
        jitter_ratio: Fraction of base delay to use as random jitter range.
            0.5 means jitter is uniform in [0, 0.5 * computed_delay].

    Returns:
        Delay in seconds: min(base * 2^(attempt-1), max_delay) + random jitter.

    The jitter decorrelates concurrent retries (e.g. multiple sessions hitting
    the same rate-limited provider).  Modeled after claw-code's splitmix64
    approach but adapted for Python's random module.
    """
    global _jitter_counter
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    # Exponential backoff: base, base*2, base*4, ... capped at max_delay.
    # Guard against overflow for extreme attempt values by short-circuiting
    # when the exponent alone would exceed what we need.
    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        # 2^63 overflows or base_delay=0 means infinite tight loop — just cap
        delay = max_delay
    else:
        computed = base_delay * (2 ** exponent)
        delay = min(computed, max_delay)

    # Additive jitter: uniform in [0, jitter_ratio * delay]
    # Seed from time + counter for decorrelation even with coarse clocks
    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    rng = random.Random(seed)
    jitter = rng.uniform(0, jitter_ratio * delay)

    return delay + jitter


def extract_retry_after(error: Exception) -> Optional[float]:
    """Extract Retry-After delay (seconds) from an exception, if present.

    Checks:
    1. error.response.headers['Retry-After'] (HTTP header)
    2. error.body JSON field 'retry_after' (some providers)
    3. Error message text ('retry after N seconds')

    Returns seconds to wait, or None if no Retry-After hint found.
    """
    # Check HTTP response headers
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) if response else None
    if headers and hasattr(headers, "get"):
        for key in ("retry-after", "Retry-After"):
            val = headers.get(key)
            if val:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    logger.debug("Could not parse %s header value as float: %r", key, val)
        # Some providers use x-ratelimit-reset (epoch timestamp)
        reset = headers.get("x-ratelimit-reset")
        if reset:
            try:
                remaining = float(reset) - time.time()
                return max(0.0, remaining)
            except (TypeError, ValueError):
                logger.debug("Could not parse x-ratelimit-reset header value as float: %r", reset)

    # Check error body JSON
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        retry_after = body.get("retry_after")
        if retry_after is not None:
            try:
                return float(retry_after)
            except (TypeError, ValueError):
                logger.debug("Could not parse body retry_after as float: %r", retry_after)

    # Parse from error message text
    msg = str(error)
    match = _RETRY_AFTER_TEXT_RE.search(msg)
    if match:
        return float(match.group(1))

    return None
