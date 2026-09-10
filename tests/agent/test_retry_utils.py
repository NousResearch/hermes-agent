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










def _zai_quota_wall_error():
    """Z.AI's OTHER 429: the plan's window is spent. Must keep failing fast."""
    return SimpleNamespace(
        status_code=429,
        body={
            "error": {
                "code": "1113",
                "message": (
                    "Weekly/Monthly Limit Exhausted. Your limit will reset at "
                    "2026-08-31 01:47:59"
                ),
            }
        },
    )


def _zai_request_rate_error():
    """Z.AI code 1302: too many requests per unit time, not service overload."""
    return SimpleNamespace(
        status_code=429,
        body={"error": {"code": "1302", "message": "Rate limit reached for requests"}},
    )


# ---------------------------------------------------------------------------
# is_zai_coding_overload_error — the predicate that gates the long-backoff tier
# ---------------------------------------------------------------------------


def test_the_historical_coding_plan_shape_still_matches():
    """The shape the policy was written for, kept as a regression."""
    assert is_zai_coding_overload_error(
        base_url="https://api.z.ai/api/coding/paas/v4",
        model="glm-5.2",
        error=_zai_overload_error(),
    )


def test_the_same_overload_on_another_model_and_path_matches():
    """The shape belongs to the SERVICE, not to one model on one path.

    Observed on a live board: workers on ``api.z.ai/api/paas/v4`` running
    ``glm-4.7-flash`` take the identical 429/1305. While the predicate was
    pinned to the coding path AND the literal ``glm-5.2``, those runs got three
    short retries over ~15s and gave up, with the entire 30/60/90/120s schedule
    unreachable for them.
    """
    assert is_zai_coding_overload_error(
        base_url="https://api.z.ai/api/paas/v4/",
        model="glm-4.7-flash",
        error=_zai_overload_error(),
    )


def test_a_quota_wall_is_not_an_overload():
    """The assertion that keeps this from swallowing the exhausted-plan case.

    A spent weekly/monthly window does not get better in 120 seconds — it must
    fail fast so the chain falls over to another rail.
    """
    assert not is_zai_coding_overload_error(
        base_url="https://api.z.ai/api/paas/v4/",
        model="glm-5.3",
        error=_zai_quota_wall_error(),
    )


def test_a_request_rate_429_is_not_an_overload():
    assert not is_zai_coding_overload_error(
        base_url="https://api.z.ai/api/paas/v4/",
        model="glm-4.7-flash",
        error=_zai_request_rate_error(),
    )


def test_another_providers_overload_keeps_its_own_policy():
    """Host-scoped: the long schedule is tuned to Z.AI's windows, not everyone's."""
    assert not is_zai_coding_overload_error(
        base_url="https://api.openai.com/v1",
        model="gpt-5.6-terra",
        error=_zai_overload_error(),
    )


def test_a_lookalike_host_does_not_match():
    """``"z.ai" in base_url`` would hand a stranger's endpoint a 2-minute wait."""
    for base in (
        "https://api.z.ai.example.com/v1",
        "https://notz.ai/v1",
        "https://z.ai.attacker.test/api/paas/v4",
    ):
        assert not is_zai_coding_overload_error(
            base_url=base, model="glm-4.7-flash", error=_zai_overload_error()
        ), base


def test_a_base_url_without_a_scheme_still_matches():
    """Call sites are not guaranteed to pass one."""
    assert is_zai_coding_overload_error(
        base_url="api.z.ai/api/paas/v4",
        model="glm-4.7-flash",
        error=_zai_overload_error(),
    )


def test_a_missing_base_url_does_not_match():
    for base in (None, "", "   "):
        assert not is_zai_coding_overload_error(
            base_url=base, model="glm-4.7-flash", error=_zai_overload_error()
        ), repr(base)


def test_a_non_429_with_overload_text_does_not_match():
    """The status is part of the shape; a 500 saying 'overloaded' retries normally."""
    err = SimpleNamespace(
        status_code=500,
        body={"error": {"code": "1305", "message": "The service may be temporarily overloaded"}},
    )
    assert not is_zai_coding_overload_error(
        base_url="https://api.z.ai/api/paas/v4/", model="glm-4.7-flash", error=err
    )


def test_the_long_tier_is_reachable_for_the_standard_endpoint(monkeypatch):
    """The end-to-end consequence of the widening, on the shape that was stuck."""
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
    from agent.retry_utils import zai_coding_overload_retry_ceiling

    err = _zai_overload_error()
    long_waits = []
    for attempt in range(1, zai_coding_overload_retry_ceiling()):
        _wait, policy = adaptive_rate_limit_backoff(
            attempt,
            base_url="https://api.z.ai/api/paas/v4/",
            model="glm-4.7-flash",
            error=err,
            default_wait=1.0,
        )
        if policy == "zai_coding_overload_long":
            long_waits.append(_wait)

    assert long_waits == [30.0, 60.0, 90.0, 120.0]


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
