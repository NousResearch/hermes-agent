"""Tests for agent.retry_utils — jittered backoff, retryable classification, Retry-After extraction."""

import time
from types import SimpleNamespace

from agent.retry_utils import (
    jittered_backoff,
    is_retryable_status,
    is_transient_transport_error,
    extract_retry_after,
    RETRYABLE_STATUS_CODES,
)


# ---------------------------------------------------------------------------
# is_retryable_status
# ---------------------------------------------------------------------------


def test_retryable_status_codes():
    """All standard transient codes should be retryable."""
    for code in (408, 413, 429, 500, 502, 503, 504, 529):
        assert is_retryable_status(code), f"{code} should be retryable"


def test_non_retryable_status_codes():
    """Client errors (except 408/413/429) should NOT be retryable."""
    for code in (400, 401, 403, 404, 422):
        assert not is_retryable_status(code), f"{code} should not be retryable"


def test_none_status_is_not_retryable():
    assert not is_retryable_status(None)


# ---------------------------------------------------------------------------
# is_transient_transport_error
# ---------------------------------------------------------------------------


def test_transient_transport_errors():
    """Common transport exceptions should be flagged as transient."""
    for msg in [
        "ReadTimeout: connection pool timed out",
        "ConnectError: connection refused",
        "RemoteProtocolError: peer closed connection",
        "network connection lost",
    ]:
        err = Exception(msg)
        assert is_transient_transport_error(err), f"'{msg}' should be transient"


def test_non_transient_errors():
    """Auth and validation errors should NOT be transient."""
    for msg in ["Invalid API key", "400 Bad Request", "model not found"]:
        err = Exception(msg)
        assert not is_transient_transport_error(err), f"'{msg}' should not be transient"


# ---------------------------------------------------------------------------
# jittered_backoff
# ---------------------------------------------------------------------------


def test_backoff_is_exponential():
    """Base delay should double each attempt (before jitter)."""
    # Run many samples and check that the mean is close to the expected value.
    # With jitter_ratio=0, there should be no jitter.
    for attempt in (1, 2, 3, 4):
        delays = [jittered_backoff(attempt, base_delay=5.0, max_delay=120.0, jitter_ratio=0.0) for _ in range(100)]
        expected = min(5.0 * (2 ** (attempt - 1)), 120.0)
        mean = sum(delays) / len(delays)
        assert abs(mean - expected) < 0.01, f"attempt {attempt}: expected {expected}, got {mean}"


def test_backoff_respects_max_delay():
    """Even with high attempt numbers, delay should not exceed max_delay."""
    for attempt in (10, 20, 100):
        delay = jittered_backoff(attempt, base_delay=5.0, max_delay=60.0, jitter_ratio=0.0)
        assert delay <= 60.0, f"attempt {attempt}: delay {delay} exceeds max 60s"


def test_backoff_adds_jitter():
    """With jitter enabled, delays should vary across calls."""
    delays = [jittered_backoff(1, base_delay=10.0, max_delay=120.0, jitter_ratio=0.5) for _ in range(50)]
    # At least some variation should exist
    assert min(delays) != max(delays), "jitter should produce varying delays"
    # All delays should be >= base (jitter is additive)
    assert all(d >= 10.0 for d in delays), "jittered delay should be >= base delay"
    # No delay should exceed base + jitter_range + small epsilon
    assert all(d <= 10.0 + 5.0 + 0.01 for d in delays), "jittered delay should be bounded"


def test_backoff_attempt_1_is_base():
    """First attempt delay should equal base_delay (with no jitter)."""
    delay = jittered_backoff(1, base_delay=3.0, max_delay=120.0, jitter_ratio=0.0)
    assert delay == 3.0


# ---------------------------------------------------------------------------
# extract_retry_after
# ---------------------------------------------------------------------------


def test_extract_retry_after_from_headers():
    """Should parse Retry-After from HTTP response headers."""
    response = SimpleNamespace(headers={"Retry-After": "30"})
    error = Exception("Rate limited")
    error.response = response  # type: ignore[attr-defined]
    assert extract_retry_after(error) == 30.0


def test_extract_retry_after_from_lowercase_headers():
    """Should parse retry-after (lowercase) from headers."""
    response = SimpleNamespace(headers={"retry-after": "45"})
    error = Exception("Rate limited")
    error.response = response  # type: ignore[attr-defined]
    assert extract_retry_after(error) == 45.0


def test_extract_retry_after_from_message():
    """Should parse 'retry after N seconds' from error message."""
    error = Exception("Error code: 429 - Rate limit. Retry after 30 seconds.")
    assert extract_retry_after(error) == 30.0


def test_extract_retry_after_from_body():
    """Should parse retry_after from error body dict."""
    error = Exception("Rate limited")
    error.body = {"retry_after": 15}  # type: ignore[attr-defined]
    assert extract_retry_after(error) == 15.0


def test_extract_retry_after_none_when_absent():
    """Should return None when no Retry-After hint exists."""
    error = Exception("Some other error")
    assert extract_retry_after(error) is None


# ---------------------------------------------------------------------------
# Edge cases (from QC review)
# ---------------------------------------------------------------------------


def test_backoff_with_zero_base_delay_returns_max():
    """base_delay=0 should return max_delay (guard against busy-wait)."""
    delay = jittered_backoff(1, base_delay=0.0, max_delay=60.0, jitter_ratio=0.0)
    assert delay == 60.0


def test_backoff_with_extreme_attempt_returns_max():
    """Very large attempt numbers should not overflow — return max_delay."""
    delay = jittered_backoff(999, base_delay=5.0, max_delay=120.0, jitter_ratio=0.0)
    assert delay == 120.0


def test_backoff_negative_attempt_treated_as_one():
    """Negative attempt should not crash, behaves like attempt=1."""
    delay = jittered_backoff(-5, base_delay=10.0, max_delay=120.0, jitter_ratio=0.0)
    assert delay == 10.0


def test_backoff_thread_safety():
    """Concurrent calls should not produce identical seeds (no race condition)."""
    import threading
    results = []
    barrier = threading.Barrier(8)

    def _call_backoff():
        barrier.wait()  # synchronize all threads
        results.append(jittered_backoff(1, base_delay=10.0, max_delay=120.0, jitter_ratio=0.5))

    threads = [threading.Thread(target=_call_backoff) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(results) == 8
    # With proper thread-safe counter, seeds should be unique
    # (not guaranteed but overwhelmingly likely with 8 threads)
    unique = len(set(results))
    assert unique >= 6, f"Expected mostly unique delays, got {unique}/8 unique"


def test_extract_retry_after_no_response_attr():
    """Should not crash when exception has no .response attribute."""
    error = Exception("plain error")
    assert extract_retry_after(error) is None


def test_extract_retry_after_empty_headers():
    """Should return None with empty headers dict."""
    response = SimpleNamespace(headers={})
    error = Exception("Rate limited")
    error.response = response  # type: ignore[attr-defined]
    assert extract_retry_after(error) is None
