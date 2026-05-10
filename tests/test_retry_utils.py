"""Tests for agent.retry_utils jittered backoff."""

import threading

import agent.retry_utils as retry_utils
from agent.retry_utils import jittered_backoff


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
