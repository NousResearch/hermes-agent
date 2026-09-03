"""Tests for agent.retry_utils jittered backoff."""

import threading

import agent.retry_utils as retry_utils
from types import SimpleNamespace

from agent.retry_utils import (
    adaptive_rate_limit_backoff,
    is_upstream_capacity_error,
    is_zai_coding_overload_error,
    jittered_backoff,
    upstream_capacity_backoff,
    upstream_capacity_retry_ceiling,
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




def test_backoff_attempt_1_is_base():
    """First attempt delay should equal base_delay (with no jitter)."""
    delay = jittered_backoff(1, base_delay=3.0, max_delay=120.0, jitter_ratio=0.0)
    assert delay == 3.0








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


# ---------------------------------------------------------------------------
# parse_retry_after_seconds — shared Retry-After parser
# ---------------------------------------------------------------------------


class TestParseRetryAfterSeconds:
    def test_numeric_string(self):
        from agent.retry_utils import parse_retry_after_seconds
        assert parse_retry_after_seconds("120") == 120.0
        assert parse_retry_after_seconds(" 4.5 ") == 4.5

    def test_numeric_value(self):
        from agent.retry_utils import parse_retry_after_seconds
        assert parse_retry_after_seconds(45) == 45.0
        assert parse_retry_after_seconds(3.25) == 3.25


    def test_http_date(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime
        from agent.retry_utils import parse_retry_after_seconds

        future = datetime.now(timezone.utc) + timedelta(seconds=90)
        seconds = parse_retry_after_seconds(format_datetime(future, usegmt=True))
        assert seconds is not None and 80 <= seconds <= 91

        past = datetime.now(timezone.utc) - timedelta(seconds=90)
        assert parse_retry_after_seconds(format_datetime(past, usegmt=True)) == 0.0



    def test_headers_get_raises(self):
        from agent.retry_utils import parse_retry_after_seconds

        class Explosive:
            def get(self, _key):
                raise RuntimeError("boom")

        assert parse_retry_after_seconds(Explosive()) is None


# ---------------------------------------------------------------------------
# Upstream-capacity (model "temporarily at capacity") backoff
# ---------------------------------------------------------------------------


def _capacity_error(msg="The requested model is temporarily at capacity upstream. This is not your API key's rate limit — please retry shortly."):
    """Build a SimpleNamespace error mimicking a Nous Portal 429 capacity error."""
    return SimpleNamespace(
        status_code=429,
        body={"error": {"message": msg}},
    )


class TestIsUpstreamCapacityError:
    def test_nous_capacity_message_is_detected(self):
        assert is_upstream_capacity_error(error=_capacity_error()) is True

    def test_generic_at_capacity_429_is_not_detected(self):
        # Generic "at capacity" / "over capacity" on a 429 without the exact
        # Nous Portal phrasing must NOT trigger the patient upstream-capacity
        # backoff — those are likely account-level quota/credential limits
        # that need a failover, not a multi-minute retry.
        err = SimpleNamespace(
            status_code=429,
            body={"error": {"message": "The server is at capacity"}},
        )
        assert is_upstream_capacity_error(error=err) is False

    def test_plain_rate_limit_is_not_detected(self):
        err = SimpleNamespace(
            status_code=429,
            body={"error": {"message": "rate limit exceeded"}},
        )
        assert is_upstream_capacity_error(error=err) is False

    def test_non_429_status_is_not_detected(self):
        err = SimpleNamespace(
            status_code=503,
            body={"error": {"message": "at capacity"}},
        )
        assert is_upstream_capacity_error(error=err) is False

    def test_no_status_code_is_not_detected(self):
        err = SimpleNamespace(
            body={"error": {"message": "temporarily at capacity upstream"}},
        )
        assert is_upstream_capacity_error(error=err) is False


class TestUpstreamCapacityRetryCeiling:
    def test_ceiling_exceeds_default_max_retries(self):
        # Default api_max_retries is 3; the upstream-capacity ceiling must be
        # strictly larger so the long backoff tier is actually reachable.
        ceiling = upstream_capacity_retry_ceiling()
        assert ceiling > 3

    def test_ceiling_covers_all_long_tier_entries(self):
        # The loop gives up at retry_count >= ceiling, so ceiling - 1 is the
        # last attempt that gets backoff. Every long-tier entry must be
        # reachable.
        from agent.retry_utils import _UPSTREAM_CAPACITY_LONG_BACKOFF, _UPSTREAM_CAPACITY_SHORT_ATTEMPTS

        ceiling = upstream_capacity_retry_ceiling()
        last_attempt_with_backoff = ceiling - 1
        assert last_attempt_with_backoff - _UPSTREAM_CAPACITY_SHORT_ATTEMPTS >= len(_UPSTREAM_CAPACITY_LONG_BACKOFF)


class TestUpstreamCapacityBackoff:
    def test_short_tier_is_exponential(self):
        # First 3 attempts use jittered_backoff with base_delay=2.0.
        # jitter_ratio=0.5 means jitter is uniform in [0, 0.5*delay],
        # so the mean is base + 0.25*base = 1.25*base.
        for attempt in (1, 2, 3):
            delays = [
                upstream_capacity_backoff(attempt)
                for _ in range(500)
            ]
            mean = sum(delays) / len(delays)
            base = min(2.0 * (2 ** (attempt - 1)), 60.0)
            # Mean should be ~1.25 * base (base + half the jitter range).
            assert abs(mean - base * 1.25) < 0.5, (
                f"attempt {attempt}: short-tier mean {mean:.1f} vs expected {base * 1.25:.1f}"
            )

    def test_long_tier_grows_progressively(self):
        # Long tier (attempts 4+) walks 10, 30, 60, 120, 180, 300 with small jitter.
        # jitter_ratio=0.2 means jitter is uniform in [0, 0.2*base],
        # so the mean is base + 0.1*base = 1.1*base.
        means = []
        for attempt in range(4, 10):
            delays = [
                upstream_capacity_backoff(attempt)
                for _ in range(500)
            ]
            means.append(sum(delays) / len(delays))

        # Each long-tier mean should be ~1.1 * base.
        expected_long = [10.0, 30.0, 60.0, 120.0, 180.0, 300.0]
        for mean, exp in zip(means, expected_long):
            assert abs(mean - exp * 1.1) < 2.0, (
                f"long-tier mean {mean:.1f} vs expected {exp * 1.1:.1f}"
            )

    def test_long_tier_stays_within_jitter_bounds(self):
        # With 0.2 jitter ratio, each long-tier value is in [base, base + 0.2*base].
        for attempt in range(4, 10):
            delays = [
                upstream_capacity_backoff(attempt)
                for _ in range(500)
            ]
            min_d = min(delays)
            max_d = max(delays)
            idx = attempt - 4
            expected_base = [10.0, 30.0, 60.0, 120.0, 180.0, 300.0][idx]
            assert min_d >= expected_base, (
                f"attempt {attempt}: min {min_d:.1f} below base {expected_base}"
            )
            assert max_d <= expected_base * 1.2, (
                f"attempt {attempt}: max {max_d:.1f} above cap {expected_base * 1.2:.1f}"
            )

    def test_ceiling_minus_1_reaches_final_long_tier(self):
        # The last attempt that gets backoff should hit the 300s tier.
        ceiling = upstream_capacity_retry_ceiling()
        last_delay = upstream_capacity_backoff(ceiling - 1)
        # 300s base with 0.2 jitter → [300, 360]
        assert 295 < last_delay <= 360, f"last delay {last_delay:.1f} should be ~300s"

    def test_call_site_retry_count_is_zero_based(self):
        # The retry loop in conversation_loop.py passes retry_count (0-based)
        # to upstream_capacity_backoff as retry_count + 1 (1-based).  Verify
        # that the mapping produces the intended schedule: the first call-site
        # retry (retry_count=0) maps to attempt 1 (short tier), and the fourth
        # call-site retry (retry_count=3) maps to attempt 4 (first long tier
        # entry, 10s base) — NOT a silent negative-index 300s stall.
        from agent.retry_utils import (
            _UPSTREAM_CAPACITY_SHORT_ATTEMPTS,
            _UPSTREAM_CAPACITY_LONG_BACKOFF,
        )
        # retry_count=0 → attempt=1 (short tier, attempt <= SHORT_ATTEMPTS)
        assert upstream_capacity_backoff(0 + 1) <= 60.0 * 1.2
        # retry_count=3 → attempt=4 (first long-tier entry: 10s base)
        delay = upstream_capacity_backoff(3 + 1)
        assert 10.0 <= delay <= 10.0 * 1.2, (
            f"retry_count=3 should map to long-tier base 10s, got {delay:.1f}s"
        )
        # retry_count=_SHORT_ATTEMPTS+len(long)-1 → attempt=ceiling-1 (300s)
        ceiling = upstream_capacity_retry_ceiling()
        last = upstream_capacity_backoff(ceiling - 1 + 1)
        assert 295 < last <= 360, f"final long-tier delay {last:.1f} should be ~300s"

    def test_negative_index_guard_prevents_silent_300s_stall(self):
        # Defense-in-depth: if upstream_capacity_backoff receives an attempt
        # below _UPSTREAM_CAPACITY_SHORT_ATTEMPTS (e.g. attempt=0 from a
        # caller that forgets the +1), the index into the long table must not
        # go negative and silently resolve to the last entry (300s).  It should
        # fall into the short tier instead.
        delay = upstream_capacity_backoff(0)
        assert delay <= 60.0 * 1.2, (
            f"attempt=0 should hit short tier, not 300s long tier; got {delay:.1f}s"
        )
