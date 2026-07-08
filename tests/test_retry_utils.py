"""Tests for agent.retry_utils jittered backoff."""

import threading

import agent.retry_utils as retry_utils
from types import SimpleNamespace

from agent.retry_utils import adaptive_rate_limit_backoff, is_zai_coding_overload_error, jittered_backoff


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


# --- resolve_retry_after: honor Retry-After on rate-limit AND overload -------
# SPEC 2026-07-07 (pool-at-capacity transient-503, 4b). The relay emits a bounded
# Retry-After on a pool-at-capacity 503; the harness honors it on the overloaded
# path (not just rate-limit), with a final-pre-fallback-retry carve-out.
from agent.retry_utils import (  # noqa: E402
    resolve_retry_after,
    RETRY_AFTER_CAP_OVERLOAD_S,
    RETRY_AFTER_CAP_RATE_LIMIT_S,
)


def _honor(**kw):
    base = dict(raw_value="8", is_rate_limit=False, is_overload=True,
                retry_count=0, max_retries=3)
    base.update(kw)
    return resolve_retry_after(**base)


def test_retry_after_honored_on_overload():
    # AC-3: an overload with a numeric Retry-After is honored (not None/jitter).
    assert _honor(raw_value="8") == 8.0


def test_retry_after_honored_on_rate_limit():
    assert _honor(is_rate_limit=True, is_overload=False, raw_value="12") == 12.0


def test_retry_after_ignored_for_other_reasons():
    # AC-4-adjacent: neither rate-limit nor overload -> never honored (jitter).
    assert _honor(is_rate_limit=False, is_overload=False, raw_value="8") is None


def test_retry_after_overload_capped_at_60():
    # INV-4: overload honors a tighter 60s cap.
    assert _honor(raw_value="999") == RETRY_AFTER_CAP_OVERLOAD_S == 60.0


def test_retry_after_rate_limit_capped_at_600():
    assert _honor(is_rate_limit=True, is_overload=False,
                  raw_value="99999") == RETRY_AFTER_CAP_RATE_LIMIT_S == 600.0


def test_retry_after_not_honored_on_final_pre_fallback_retry():
    # AC-8 / RC-1: reserve the LAST reachable retry for a fast jitter→fallback,
    # but ONLY when there are ≥2 reachable retries (max_retries ≥ 3). The caller
    # is 1-based (retry_count incremented before this runs) and activates
    # fallback at retry_count >= max_retries, so reachable retries are 1..N-1.
    # max_retries=3 -> reachable {1,2}: honor 1, jitter 2 (the last reachable).
    assert _honor(retry_count=1, max_retries=3) == 8.0   # 1 < 3-1=2 -> honor
    assert _honor(retry_count=2, max_retries=3) is None  # last reachable -> jitter
    # boundary: honor iff retry_count < max_retries-1 (when max_retries >= 3)
    assert _honor(retry_count=3, max_retries=5) == 8.0   # 3 < 5-1=4 -> honor
    assert _honor(retry_count=4, max_retries=5) is None  # last reachable -> jitter


def test_retry_after_honored_when_only_one_reachable_retry():
    # Greptile #223 P1: with max_retries=2 there is exactly ONE reachable retry
    # (retry_count=1); reserving it for jitter would make the whole overload
    # feature a no-op. So it MUST be honored.
    assert _honor(retry_count=1, max_retries=2) == 8.0


def test_retry_after_max_retries_1_never_honors():
    # max_retries=1: zero reachable retries (retry_count 1 >= max_retries 1) ->
    # never honored (straight to fallback).
    assert _honor(retry_count=1, max_retries=1) is None


def test_retry_after_http_date_falls_through_to_jitter():
    # A RFC-valid HTTP-date Retry-After is not float()-parseable -> None (jitter),
    # not a crash. Covers the pass-2 security-lens HTTP-date path.
    assert _honor(raw_value="Wed, 21 Oct 2026 07:28:00 GMT") is None


def test_retry_after_garbage_and_missing_fall_through():
    assert _honor(raw_value="abc") is None
    assert _honor(raw_value=None) is None
    assert _honor(raw_value="") is None


def test_retry_after_nonpositive_falls_through():
    assert _honor(raw_value="0") is None
    assert _honor(raw_value="-5") is None
