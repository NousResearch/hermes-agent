"""Tests for agent.retry_utils jittered backoff."""

import threading

import agent.retry_utils as retry_utils
from types import SimpleNamespace

from agent.retry_utils import (
    adaptive_rate_limit_backoff,
    is_nvidia_nim_rate_limit_error,
    is_zai_coding_overload_error,
    jittered_backoff,
)


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
    import time

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


def _zai_overload_error():
    return SimpleNamespace(
        status_code=429,
        body={
            "error": {
                "code": "1305",
                "message": "The service may be temporarily overloaded, please try again later",
            }
        },
    )


def test_zai_coding_overload_classifier_is_narrow():
    err = _zai_overload_error()
    assert is_zai_coding_overload_error(
        base_url="https://api.z.ai/api/coding/paas/v4",
        model="glm-5.2",
        error=err,
    )

    assert not is_zai_coding_overload_error(
        base_url="https://api.z.ai/api/paas/v4",
        model="glm-5.2",
        error=err,
    )
    assert not is_zai_coding_overload_error(
        base_url="https://api.z.ai/api/coding/paas/v4",
        model="glm-5.1",
        error=err,
    )
    assert not is_zai_coding_overload_error(
        base_url="https://api.z.ai/api/coding/paas/v4",
        model="glm-5.2",
        error=SimpleNamespace(status_code=429, body={"error": {"code": "1113", "message": "Insufficient balance"}}),
    )


def test_zai_coding_overload_backoff_keeps_first_retries_short(monkeypatch):
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
    err = _zai_overload_error()

    wait, policy = adaptive_rate_limit_backoff(
        1,
        base_url="https://api.z.ai/api/coding/paas/v4",
        model="glm-5.2",
        error=err,
        default_wait=2.5,
    )
    assert wait == 2.5
    assert policy == "zai_coding_overload_short"

    wait, policy = adaptive_rate_limit_backoff(
        3,
        base_url="https://api.z.ai/api/coding/paas/v4",
        model="glm-5.2",
        error=err,
        default_wait=9.0,
    )
    assert wait == 9.0
    assert policy == "zai_coding_overload_short"


def test_zai_coding_overload_backoff_grows_after_short_retries(monkeypatch):
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
    err = _zai_overload_error()

    waits = []
    for attempt in range(4, 10):
        wait, policy = adaptive_rate_limit_backoff(
            attempt,
            base_url="https://api.z.ai/api/coding/paas/v4",
            model="glm-5.2",
            error=err,
            default_wait=10.0,
        )
        waits.append(wait)
        assert policy == "zai_coding_overload_long"

    assert waits == [30.0, 60.0, 90.0, 120.0, 120.0, 120.0]


def test_nvidia_nim_rate_limit_classifier_is_endpoint_specific():
    err = SimpleNamespace(status_code=429, body={"status": 429, "title": "Too Many Requests"})

    assert is_nvidia_nim_rate_limit_error(
        base_url="https://integrate.api.nvidia.com/v1",
        error=err,
    )
    assert is_nvidia_nim_rate_limit_error(
        base_url="https://integrate.api.nvidia.com/v1/",
        error=SimpleNamespace(body='{"status":429,"title":"Too Many Requests"}'),
    )
    assert not is_nvidia_nim_rate_limit_error(
        base_url="https://openrouter.ai/api/v1",
        error=err,
    )


def test_nvidia_nim_rate_limit_backoff_uses_wider_window(monkeypatch):
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
    err = SimpleNamespace(status_code=429, body={"status": 429, "title": "Too Many Requests"})

    waits = []
    for attempt in range(1, 6):
        wait, policy = adaptive_rate_limit_backoff(
            attempt,
            base_url="https://integrate.api.nvidia.com/v1",
            model="minimaxai/minimax-m3",
            error=err,
            default_wait=2.0,
        )
        waits.append(wait)
        assert policy == "nvidia_nim_rate_limit"

    assert waits == [15.0, 30.0, 60.0, 120.0, 120.0]


def test_non_zai_backoff_returns_default_wait():
    wait, policy = adaptive_rate_limit_backoff(
        10,
        base_url="https://openrouter.ai/api/v1",
        model="glm-5.2",
        error=_zai_overload_error(),
        default_wait=12.0,
    )
    assert wait == 12.0
    assert policy is None


def test_zai_overload_retry_ceiling_exceeds_short_attempts():
    """Invariant: the ceiling must sit above the short-retry threshold, or the
    long-backoff tier is unreachable and the whole schedule is dead code
    (the original bug: default api_max_retries == short_attempts == 3)."""
    from agent.retry_utils import (
        zai_coding_overload_retry_ceiling,
        _ZAI_CODING_OVERLOAD_LONG_BACKOFF,
    )

    short_attempts = 3
    ceiling = zai_coding_overload_retry_ceiling(short_attempts)
    assert ceiling > short_attempts
    # Invariant (not a formula mirror): the loop's give-up check
    # (retry_count >= ceiling) runs *before* the attempt's backoff, so the
    # ceiling must leave headroom for every long-backoff entry to execute —
    # i.e. the largest attempt the loop still computes backoff for
    # (ceiling - 1) must reach the final long-tier index.
    last_attempt_with_backoff = ceiling - 1
    assert last_attempt_with_backoff - short_attempts >= len(_ZAI_CODING_OVERLOAD_LONG_BACKOFF)


def test_zai_overload_ceiling_makes_long_tier_reachable(monkeypatch):
    """End-to-end over the attempt range the retry loop actually walks: with the
    extended ceiling, at least one attempt reaches the long-backoff tier and the
    full 30/60/90/120s schedule is exercised."""
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
    from agent.retry_utils import zai_coding_overload_retry_ceiling

    err = _zai_overload_error()
    ceiling = zai_coding_overload_retry_ceiling()

    long_waits = []
    # The loop computes backoff for attempts 1..ceiling-1 (it gives up at ceiling).
    for attempt in range(1, ceiling):
        _wait, policy = adaptive_rate_limit_backoff(
            attempt,
            base_url="https://api.z.ai/api/coding/paas/v4",
            model="glm-5.2",
            error=err,
            default_wait=1.0,
        )
        if policy == "zai_coding_overload_long":
            long_waits.append(_wait)

    assert long_waits, "long-backoff tier never reached within the retry ceiling"
    assert long_waits == [30.0, 60.0, 90.0, 120.0]
