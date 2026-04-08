"""Tests for agent.retry_utils retry helpers."""

import threading
import time
from types import SimpleNamespace

import agent.retry_utils as retry_utils
from agent.retry_utils import (
    extract_retry_after,
    is_retryable_status,
    is_transient_transport_error,
    jittered_backoff,
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
        assert is_transient_transport_error(err), f"{msg!r} should be transient"


def test_non_transient_errors():
    """Auth and validation errors should NOT be transient."""
    for msg in ["Invalid API key", "400 Bad Request", "model not found"]:
        err = Exception(msg)
        assert not is_transient_transport_error(err), f"{msg!r} should not be transient"


def test_transient_transport_error_is_case_insensitive():
    err = Exception("NeTwOrK CoNnEcTiOn LoSt")
    assert is_transient_transport_error(err)


# ---------------------------------------------------------------------------
# jittered_backoff
# ---------------------------------------------------------------------------


def test_backoff_is_exponential():
    """Base delay should double each attempt (before jitter)."""
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
    assert min(delays) != max(delays), "jitter should produce varying delays"
    assert all(d >= 10.0 for d in delays), "jittered delay should be >= base delay"
    assert all(d <= 15.0 for d in delays), "jittered delay should be bounded"


def test_backoff_attempt_1_is_base():
    """First attempt delay should equal base_delay (with no jitter)."""
    delay = jittered_backoff(1, base_delay=3.0, max_delay=120.0, jitter_ratio=0.0)
    assert delay == 3.0


def test_backoff_with_zero_base_delay_returns_max():
    """base_delay=0 should return max_delay (guard against busy-wait)."""
    delay = jittered_backoff(1, base_delay=0.0, max_delay=60.0, jitter_ratio=0.0)
    assert delay == 60.0


def test_backoff_with_extreme_attempt_returns_max():
    """Very large attempt numbers should not overflow and should return max_delay."""
    delay = jittered_backoff(999, base_delay=5.0, max_delay=120.0, jitter_ratio=0.0)
    assert delay == 120.0


def test_backoff_negative_attempt_treated_as_one():
    """Negative attempt should not crash and behaves like attempt=1."""
    delay = jittered_backoff(-5, base_delay=10.0, max_delay=120.0, jitter_ratio=0.0)
    assert delay == 10.0


def test_backoff_thread_safety():
    """Concurrent calls should generally produce different delays."""
    results = []
    barrier = threading.Barrier(8)

    def _call_backoff():
        barrier.wait()
        results.append(jittered_backoff(1, base_delay=10.0, max_delay=120.0, jitter_ratio=0.5))

    threads = [threading.Thread(target=_call_backoff) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(results) == 8
    unique = len(set(results))
    assert unique >= 6, f"Expected mostly unique delays, got {unique}/8 unique"


def test_backoff_uses_locked_tick_for_seed(monkeypatch):
    """Seed derivation should use per-call tick captured under lock."""
    monkeypatch.setattr(retry_utils, "_jitter_counter", 0)

    recorded_seeds = []

    class _RecordingRandom:
        def __init__(self, seed):
            recorded_seeds.append(seed)

        def uniform(self, a, b):
            return 0.0

    monkeypatch.setattr(retry_utils.random, "Random", _RecordingRandom)

    fixed_time_ns = 123456789

    def _time_ns_wait_for_two_ticks():
        deadline = time.time() + 2.0
        while retry_utils._jitter_counter < 2 and time.time() < deadline:
            time.sleep(0.001)
        return fixed_time_ns

    monkeypatch.setattr(retry_utils.time, "time_ns", _time_ns_wait_for_two_ticks)

    barrier = threading.Barrier(2)

    def _call():
        barrier.wait()
        jittered_backoff(1, base_delay=10.0, max_delay=120.0, jitter_ratio=0.5)

    threads = [threading.Thread(target=_call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(recorded_seeds) == 2
    assert len(set(recorded_seeds)) == 2, f"Expected unique seeds, got {recorded_seeds}"


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


def test_extract_retry_after_from_x_ratelimit_reset_future():
    """x-ratelimit-reset should return positive seconds until reset."""
    now = time.time()
    response = SimpleNamespace(headers={"x-ratelimit-reset": str(now + 20)})
    error = Exception("Rate limited")
    error.response = response  # type: ignore[attr-defined]
    retry_after = extract_retry_after(error)
    assert retry_after is not None
    assert 15.0 <= retry_after <= 20.0


def test_extract_retry_after_from_x_ratelimit_reset_past_returns_zero():
    """Past reset timestamps should translate to immediate retry (0s)."""
    now = time.time()
    response = SimpleNamespace(headers={"x-ratelimit-reset": str(now - 5)})
    error = Exception("Rate limited")
    error.response = response  # type: ignore[attr-defined]
    assert extract_retry_after(error) == 0.0


def test_extract_retry_after_from_message():
    """Should parse 'retry after N seconds' from error message."""
    error = Exception("Error code: 429 - Rate limit. Retry after 30 seconds.")
    assert extract_retry_after(error) == 30.0


def test_extract_retry_after_message_unit_variants():
    """All supported short unit variants should be parsed."""
    assert extract_retry_after(Exception("retry after 5 sec")) == 5.0
    assert extract_retry_after(Exception("retry after 6 secs")) == 6.0
    assert extract_retry_after(Exception("retry after 7s")) == 7.0


def test_extract_retry_after_from_body():
    """Should parse retry_after from error body dict."""
    error = Exception("Rate limited")
    error.body = {"retry_after": 15}  # type: ignore[attr-defined]
    assert extract_retry_after(error) == 15.0


def test_extract_retry_after_priority_header_over_body_and_message():
    response = SimpleNamespace(headers={"Retry-After": "30"})
    error = Exception("Retry after 10 seconds")
    error.response = response  # type: ignore[attr-defined]
    error.body = {"retry_after": 20}  # type: ignore[attr-defined]
    assert extract_retry_after(error) == 30.0


def test_extract_retry_after_priority_body_over_message():
    error = Exception("Retry after 10 seconds")
    error.body = {"retry_after": 20}  # type: ignore[attr-defined]
    assert extract_retry_after(error) == 20.0


def test_extract_retry_after_ignores_non_dict_body():
    error = Exception("not parseable")
    error.body = "retry_after=25"  # type: ignore[attr-defined]
    assert extract_retry_after(error) is None


def test_extract_retry_after_none_when_absent():
    """Should return None when no Retry-After hint exists."""
    error = Exception("Some other error")
    assert extract_retry_after(error) is None


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
