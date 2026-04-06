"""Tests for agent/provider_cooldown.py — provider cooldown / circuit breaker."""

import threading
import time
from unittest.mock import patch

import pytest

from agent.provider_cooldown import (
    CooldownEntry,
    ProviderCooldownTracker,
    ProviderHealthStats,
    get_cooldown_tracker,
    _backoff_seconds,
    _make_key,
    REASON_RATE_LIMIT,
    REASON_AUTH,
    REASON_AUTH_PERMANENT,
    REASON_OVERLOADED,
    REASON_BILLING,
    _TRANSIENT_BACKOFF,
    _PERMANENT_BACKOFF,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the singleton between tests to avoid cross-contamination."""
    ProviderCooldownTracker._reset_singleton()
    yield
    ProviderCooldownTracker._reset_singleton()


@pytest.fixture
def tracker():
    """Return a fresh tracker instance (via singleton)."""
    return get_cooldown_tracker()


# ---------------------------------------------------------------------------
# CooldownEntry dataclass
# ---------------------------------------------------------------------------

class TestCooldownEntry:
    def test_defaults(self):
        entry = CooldownEntry(provider_key="p::u", reason="rate_limit")
        assert entry.provider_key == "p::u"
        assert entry.reason == "rate_limit"
        assert entry.error_count == 0
        assert entry.cooldown_until == 0.0
        assert entry.last_failure_at == 0.0


# ---------------------------------------------------------------------------
# _backoff_seconds helper
# ---------------------------------------------------------------------------

class TestBackoffSeconds:
    def test_transient_escalation(self):
        assert _backoff_seconds(1, REASON_RATE_LIMIT) == 30.0
        assert _backoff_seconds(2, REASON_RATE_LIMIT) == 60.0
        assert _backoff_seconds(3, REASON_RATE_LIMIT) == 300.0
        # 4+ errors stay at max tier
        assert _backoff_seconds(5, REASON_RATE_LIMIT) == 300.0
        assert _backoff_seconds(100, REASON_RATE_LIMIT) == 300.0

    def test_overloaded_uses_transient_schedule(self):
        assert _backoff_seconds(1, REASON_OVERLOADED) == 30.0
        assert _backoff_seconds(2, REASON_OVERLOADED) == 60.0
        assert _backoff_seconds(3, REASON_OVERLOADED) == 300.0

    def test_auth_uses_transient_schedule(self):
        # Generic auth is transient (may be refreshable)
        assert _backoff_seconds(1, REASON_AUTH) == 30.0
        assert _backoff_seconds(2, REASON_AUTH) == 60.0

    def test_permanent_escalation(self):
        assert _backoff_seconds(1, REASON_AUTH_PERMANENT) == 300.0
        assert _backoff_seconds(2, REASON_AUTH_PERMANENT) == 600.0
        assert _backoff_seconds(3, REASON_AUTH_PERMANENT) == 1800.0
        assert _backoff_seconds(10, REASON_AUTH_PERMANENT) == 1800.0

    def test_billing_uses_permanent_schedule(self):
        assert _backoff_seconds(1, REASON_BILLING) == 300.0
        assert _backoff_seconds(2, REASON_BILLING) == 600.0
        assert _backoff_seconds(3, REASON_BILLING) == 1800.0

    def test_zero_error_count_clamps_to_first_tier(self):
        # Edge case: should not crash
        assert _backoff_seconds(0, REASON_RATE_LIMIT) == 30.0


# ---------------------------------------------------------------------------
# _make_key helper
# ---------------------------------------------------------------------------

class TestMakeKey:
    def test_basic(self):
        assert _make_key("openrouter", "https://api.example.com") == "openrouter::https://api.example.com"

    def test_empty_base_url(self):
        assert _make_key("anthropic", "") == "anthropic::"
        assert _make_key("anthropic", None) == "anthropic::"


# ---------------------------------------------------------------------------
# ProviderCooldownTracker
# ---------------------------------------------------------------------------

class TestRecordFailure:
    def test_first_failure_creates_entry(self, tracker):
        entry = tracker.record_failure("openrouter", "https://api.example.com", REASON_RATE_LIMIT)
        assert entry.error_count == 1
        assert entry.reason == REASON_RATE_LIMIT
        assert entry.cooldown_until > time.time()
        assert entry.last_failure_at > 0

    def test_escalating_error_count(self, tracker):
        tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        entry = tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        assert entry.error_count == 2

        entry = tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        assert entry.error_count == 3

    def test_escalating_cooldown_duration(self, tracker):
        now = time.time()

        entry1 = tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        # First failure: ~30s backoff
        assert 29 <= (entry1.cooldown_until - now) <= 32

        entry2 = tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        now2 = time.time()
        # Second failure: ~60s backoff
        assert 59 <= (entry2.cooldown_until - now2) <= 62

        entry3 = tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        now3 = time.time()
        # Third failure: ~300s backoff
        assert 299 <= (entry3.cooldown_until - now3) <= 302

    def test_reason_changes_on_subsequent_failure(self, tracker):
        tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        entry = tracker.record_failure("p", "u", REASON_OVERLOADED)
        assert entry.reason == REASON_OVERLOADED
        assert entry.error_count == 2

    def test_permanent_reason_gets_longer_backoff(self, tracker):
        now = time.time()
        entry = tracker.record_failure("p", "u", REASON_BILLING)
        # Billing first failure: ~300s
        assert 299 <= (entry.cooldown_until - now) <= 302


class TestRecordSuccess:
    def test_success_clears_entry(self, tracker):
        tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        assert tracker.is_in_cooldown("p", "u") is not None

        tracker.record_success("p", "u")
        assert tracker.is_in_cooldown("p", "u") is None

    def test_success_on_unknown_provider_is_noop(self, tracker):
        # Should not raise
        tracker.record_success("unknown", "unknown")

    def test_success_resets_error_count(self, tracker):
        tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        tracker.record_success("p", "u")

        # New failure should start at count 1 again
        entry = tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        assert entry.error_count == 1


class TestIsInCooldown:
    def test_returns_none_for_unknown_provider(self, tracker):
        assert tracker.is_in_cooldown("unknown", "url") is None

    def test_returns_entry_during_cooldown(self, tracker):
        tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        entry = tracker.is_in_cooldown("p", "u")
        assert entry is not None
        assert entry.error_count == 1

    def test_auto_clears_expired_cooldown(self, tracker):
        tracker.record_failure("p", "u", REASON_RATE_LIMIT)

        # Manually expire the cooldown
        key = _make_key("p", "u")
        with tracker._lock:
            tracker._cooldowns[key].cooldown_until = time.time() - 1

        assert tracker.is_in_cooldown("p", "u") is None

        # Entry should be removed
        with tracker._lock:
            assert key not in tracker._cooldowns

    def test_returns_entry_when_not_yet_expired(self, tracker):
        tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        entry = tracker.is_in_cooldown("p", "u")
        assert entry is not None
        assert entry.cooldown_until > time.time()


class TestClearAll:
    def test_clears_all_entries(self, tracker):
        tracker.record_failure("p1", "u1", REASON_RATE_LIMIT)
        tracker.record_failure("p2", "u2", REASON_BILLING)

        tracker.clear_all()

        assert tracker.is_in_cooldown("p1", "u1") is None
        assert tracker.is_in_cooldown("p2", "u2") is None

    def test_clear_all_on_empty_tracker(self, tracker):
        # Should not raise
        tracker.clear_all()


class TestClearProvider:
    def test_clears_specific_provider(self, tracker):
        tracker.record_failure("p1", "u1", REASON_RATE_LIMIT)
        tracker.record_failure("p2", "u2", REASON_BILLING)

        tracker.clear_provider("p1", "u1")

        assert tracker.is_in_cooldown("p1", "u1") is None
        assert tracker.is_in_cooldown("p2", "u2") is not None

    def test_clear_unknown_provider_is_noop(self, tracker):
        tracker.clear_provider("unknown", "url")


class TestGetCooldownSummary:
    def test_empty_summary(self, tracker):
        assert tracker.get_cooldown_summary() == {}

    def test_summary_contains_active_entries(self, tracker):
        tracker.record_failure("p1", "u1", REASON_RATE_LIMIT)
        tracker.record_failure("p2", "u2", REASON_BILLING)

        summary = tracker.get_cooldown_summary()
        assert len(summary) == 2
        assert "p1::u1" in summary
        assert "p2::u2" in summary

        entry = summary["p1::u1"]
        assert entry["reason"] == REASON_RATE_LIMIT
        assert entry["error_count"] == 1
        assert entry["remaining_seconds"] > 0
        assert "cooldown_until" in entry
        assert "last_failure_at" in entry

    def test_summary_excludes_expired(self, tracker):
        tracker.record_failure("p1", "u1", REASON_RATE_LIMIT)

        # Expire the entry
        key = _make_key("p1", "u1")
        with tracker._lock:
            tracker._cooldowns[key].cooldown_until = time.time() - 1

        summary = tracker.get_cooldown_summary()
        assert len(summary) == 0

        # Expired entry should have been cleaned up
        with tracker._lock:
            assert key not in tracker._cooldowns


# ---------------------------------------------------------------------------
# Singleton pattern
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_instance_returns_same_object(self):
        t1 = ProviderCooldownTracker.get_instance()
        t2 = ProviderCooldownTracker.get_instance()
        assert t1 is t2

    def test_get_cooldown_tracker_returns_singleton(self):
        t1 = get_cooldown_tracker()
        t2 = get_cooldown_tracker()
        assert t1 is t2

    def test_reset_singleton_creates_new_instance(self):
        t1 = get_cooldown_tracker()
        ProviderCooldownTracker._reset_singleton()
        t2 = get_cooldown_tracker()
        assert t1 is not t2

    def test_state_isolated_after_reset(self):
        t1 = get_cooldown_tracker()
        t1.record_failure("p", "u", REASON_RATE_LIMIT)

        ProviderCooldownTracker._reset_singleton()
        t2 = get_cooldown_tracker()
        assert t2.is_in_cooldown("p", "u") is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_record_failure(self, tracker):
        """Multiple threads recording failures should not corrupt state."""
        errors = []
        barrier = threading.Barrier(10)

        def worker(provider_id):
            try:
                barrier.wait(timeout=5)
                for _ in range(50):
                    tracker.record_failure(f"p{provider_id}", "u", REASON_RATE_LIMIT)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

        # Each provider should have exactly 50 errors
        for i in range(10):
            key = _make_key(f"p{i}", "u")
            with tracker._lock:
                entry = tracker._cooldowns.get(key)
                assert entry is not None
                assert entry.error_count == 50

    def test_concurrent_mixed_operations(self, tracker):
        """Mix of record_failure, record_success, is_in_cooldown shouldn't crash."""
        errors = []
        barrier = threading.Barrier(6)

        def fail_worker():
            try:
                barrier.wait(timeout=5)
                for _ in range(100):
                    tracker.record_failure("p", "u", REASON_RATE_LIMIT)
            except Exception as e:
                errors.append(e)

        def success_worker():
            try:
                barrier.wait(timeout=5)
                for _ in range(100):
                    tracker.record_success("p", "u")
            except Exception as e:
                errors.append(e)

        def check_worker():
            try:
                barrier.wait(timeout=5)
                for _ in range(100):
                    tracker.is_in_cooldown("p", "u")
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=fail_worker))
            threads.append(threading.Thread(target=success_worker))
            threads.append(threading.Thread(target=check_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# Different reason types produce different backoff schedules
# ---------------------------------------------------------------------------

class TestReasonSchedules:
    def test_rate_limit_uses_transient_schedule(self, tracker):
        entry = tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        expected = _TRANSIENT_BACKOFF[0]
        actual = entry.cooldown_until - entry.last_failure_at
        assert abs(actual - expected) < 1.0

    def test_overloaded_uses_transient_schedule(self, tracker):
        entry = tracker.record_failure("p", "u", REASON_OVERLOADED)
        expected = _TRANSIENT_BACKOFF[0]
        actual = entry.cooldown_until - entry.last_failure_at
        assert abs(actual - expected) < 1.0

    def test_auth_uses_transient_schedule(self, tracker):
        entry = tracker.record_failure("p", "u", REASON_AUTH)
        expected = _TRANSIENT_BACKOFF[0]
        actual = entry.cooldown_until - entry.last_failure_at
        assert abs(actual - expected) < 1.0

    def test_auth_permanent_uses_permanent_schedule(self, tracker):
        entry = tracker.record_failure("p", "u", REASON_AUTH_PERMANENT)
        expected = _PERMANENT_BACKOFF[0]
        actual = entry.cooldown_until - entry.last_failure_at
        assert abs(actual - expected) < 1.0

    def test_billing_uses_permanent_schedule(self, tracker):
        entry = tracker.record_failure("p", "u", REASON_BILLING)
        expected = _PERMANENT_BACKOFF[0]
        actual = entry.cooldown_until - entry.last_failure_at
        assert abs(actual - expected) < 1.0


# ---------------------------------------------------------------------------
# Integration: multiple providers tracked independently
# ---------------------------------------------------------------------------

class TestMultiProvider:
    def test_independent_tracking(self, tracker):
        tracker.record_failure("openrouter", "https://api.openrouter.ai/v1", REASON_RATE_LIMIT)
        tracker.record_failure("anthropic", "https://api.anthropic.com", REASON_OVERLOADED)

        assert tracker.is_in_cooldown("openrouter", "https://api.openrouter.ai/v1") is not None
        assert tracker.is_in_cooldown("anthropic", "https://api.anthropic.com") is not None

        tracker.record_success("openrouter", "https://api.openrouter.ai/v1")
        assert tracker.is_in_cooldown("openrouter", "https://api.openrouter.ai/v1") is None
        assert tracker.is_in_cooldown("anthropic", "https://api.anthropic.com") is not None

    def test_same_provider_different_urls(self, tracker):
        tracker.record_failure("openai", "https://api.openai.com/v1", REASON_RATE_LIMIT)
        tracker.record_failure("openai", "https://custom-proxy.com/v1", REASON_BILLING)

        e1 = tracker.is_in_cooldown("openai", "https://api.openai.com/v1")
        e2 = tracker.is_in_cooldown("openai", "https://custom-proxy.com/v1")

        assert e1 is not None
        assert e2 is not None
        assert e1.reason == REASON_RATE_LIMIT
        assert e2.reason == REASON_BILLING


# ---------------------------------------------------------------------------
# ProviderHealthStats dataclass
# ---------------------------------------------------------------------------

class TestProviderHealthStats:
    def test_defaults(self):
        stats = ProviderHealthStats()
        assert stats.success_count == 0
        assert stats.error_count == 0
        assert stats.total_latency_ms == 0.0
        assert stats.last_error_reason is None
        assert stats.last_error_at is None
        assert stats.last_success_at is None

    def test_total_calls(self):
        stats = ProviderHealthStats(success_count=5, error_count=3)
        assert stats.total_calls == 8

    def test_error_rate_zero_calls(self):
        stats = ProviderHealthStats()
        assert stats.error_rate == 0.0

    def test_error_rate(self):
        stats = ProviderHealthStats(success_count=7, error_count=3)
        assert abs(stats.error_rate - 0.3) < 1e-9

    def test_error_rate_all_errors(self):
        stats = ProviderHealthStats(success_count=0, error_count=5)
        assert stats.error_rate == 1.0

    def test_avg_latency_no_successes(self):
        stats = ProviderHealthStats()
        assert stats.avg_latency_ms == 0.0

    def test_avg_latency(self):
        stats = ProviderHealthStats(success_count=4, total_latency_ms=2000.0)
        assert stats.avg_latency_ms == 500.0


# ---------------------------------------------------------------------------
# Health stats accumulation
# ---------------------------------------------------------------------------

class TestHealthStatsAccumulation:
    def test_success_updates_health(self, tracker):
        tracker.record_success("p", "u", latency_ms=100.0)
        stats = tracker.get_health_stats("p", "u")
        key = _make_key("p", "u")
        assert key in stats
        h = stats[key]
        assert h.success_count == 1
        assert h.error_count == 0
        assert h.total_latency_ms == 100.0
        assert h.last_success_at is not None

    def test_failure_updates_health(self, tracker):
        tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        stats = tracker.get_health_stats("p", "u")
        key = _make_key("p", "u")
        h = stats[key]
        assert h.error_count == 1
        assert h.success_count == 0
        assert h.last_error_reason == REASON_RATE_LIMIT
        assert h.last_error_at is not None

    def test_mixed_accumulation(self, tracker):
        tracker.record_success("p", "u", latency_ms=100.0)
        tracker.record_success("p", "u", latency_ms=200.0)
        tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        tracker.record_success("p", "u", latency_ms=150.0)

        key = _make_key("p", "u")
        h = tracker.get_health_stats("p", "u")[key]
        assert h.success_count == 3
        assert h.error_count == 1
        assert h.total_latency_ms == 450.0
        assert abs(h.avg_latency_ms - 150.0) < 1e-9
        assert abs(h.error_rate - 0.25) < 1e-9

    def test_success_without_latency(self, tracker):
        tracker.record_success("p", "u")
        key = _make_key("p", "u")
        h = tracker.get_health_stats("p", "u")[key]
        assert h.success_count == 1
        assert h.total_latency_ms == 0.0
        assert h.avg_latency_ms == 0.0


# ---------------------------------------------------------------------------
# Health stats survive cooldown expiry
# ---------------------------------------------------------------------------

class TestHealthStatsPersistence:
    def test_health_survives_cooldown_expiry(self, tracker):
        """Health stats persist even when cooldown entry is auto-cleared."""
        tracker.record_failure("p", "u", REASON_RATE_LIMIT)
        tracker.record_success("p", "u", latency_ms=200.0)

        # Cooldown entry is cleared by record_success
        assert tracker.is_in_cooldown("p", "u") is None

        # But health stats still exist
        key = _make_key("p", "u")
        h = tracker.get_health_stats("p", "u")[key]
        assert h.error_count == 1
        assert h.success_count == 1
        assert h.total_latency_ms == 200.0

    def test_health_survives_expired_cooldown_auto_clear(self, tracker):
        """Health stats persist when is_in_cooldown auto-clears expired entry."""
        tracker.record_failure("p", "u", REASON_RATE_LIMIT)

        # Manually expire the cooldown
        key = _make_key("p", "u")
        with tracker._lock:
            tracker._cooldowns[key].cooldown_until = time.time() - 1

        # is_in_cooldown will auto-clear the expired entry
        assert tracker.is_in_cooldown("p", "u") is None

        # Health stats should still be there
        h = tracker.get_health_stats("p", "u")[key]
        assert h.error_count == 1


# ---------------------------------------------------------------------------
# get_health_summary
# ---------------------------------------------------------------------------

class TestGetHealthSummary:
    def test_empty_summary(self, tracker):
        assert tracker.get_health_summary() == {}

    def test_summary_format(self, tracker):
        tracker.record_success("p1", "u1", latency_ms=100.0)
        tracker.record_success("p1", "u1", latency_ms=300.0)
        tracker.record_failure("p1", "u1", REASON_RATE_LIMIT)

        summary = tracker.get_health_summary()
        key = _make_key("p1", "u1")
        assert key in summary
        entry = summary[key]

        assert entry["total_calls"] == 3
        assert entry["success"] == 2
        assert entry["errors"] == 1
        assert abs(entry["error_rate"] - 0.333) < 0.01
        assert entry["avg_latency_ms"] == 200.0
        assert entry["last_error_reason"] == REASON_RATE_LIMIT

    def test_summary_multiple_providers(self, tracker):
        tracker.record_success("p1", "u1", latency_ms=100.0)
        tracker.record_failure("p2", "u2", REASON_BILLING)

        summary = tracker.get_health_summary()
        assert len(summary) == 2
        assert _make_key("p1", "u1") in summary
        assert _make_key("p2", "u2") in summary


# ---------------------------------------------------------------------------
# clear_all clears health stats
# ---------------------------------------------------------------------------

class TestClearAllHealth:
    def test_clear_all_clears_health(self, tracker):
        tracker.record_success("p1", "u1", latency_ms=100.0)
        tracker.record_failure("p2", "u2", REASON_RATE_LIMIT)

        tracker.clear_all()

        assert tracker.get_health_stats() == {}
        assert tracker.get_health_summary() == {}


# ---------------------------------------------------------------------------
# get_health_stats filtering
# ---------------------------------------------------------------------------

class TestGetHealthStatsFiltering:
    def test_no_filter_returns_all(self, tracker):
        tracker.record_success("p1", "u1")
        tracker.record_success("p2", "u2")

        stats = tracker.get_health_stats()
        assert len(stats) == 2

    def test_filter_by_provider_and_url(self, tracker):
        tracker.record_success("p1", "u1")
        tracker.record_success("p2", "u2")

        stats = tracker.get_health_stats(provider="p1", base_url="u1")
        assert len(stats) == 1
        assert _make_key("p1", "u1") in stats

    def test_filter_by_provider_only(self, tracker):
        tracker.record_success("openai", "https://api.openai.com/v1")
        tracker.record_success("openai", "https://custom-proxy.com/v1")
        tracker.record_success("anthropic", "https://api.anthropic.com")

        stats = tracker.get_health_stats(provider="openai")
        assert len(stats) == 2
        assert all(k.startswith("openai::") for k in stats)

    def test_filter_nonexistent_provider(self, tracker):
        tracker.record_success("p1", "u1")
        stats = tracker.get_health_stats(provider="nonexistent", base_url="u1")
        assert stats == {}

    def test_filter_provider_only_nonexistent(self, tracker):
        tracker.record_success("p1", "u1")
        stats = tracker.get_health_stats(provider="nonexistent")
        assert stats == {}
