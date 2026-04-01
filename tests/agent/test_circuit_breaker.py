"""Tests for ProviderCircuitBreaker — process-wide provider rate-limit circuit breaker.

Covers:
1. Threshold tripping — breaker opens after N failures
2. Retry-After timeout behavior — breaker respects retry-after headers
3. Half-open probe selection — after reset_timeout, breaker enters half_open and allows one probe
4. Success resets — successful call closes the circuit
5. Concurrent access patterns — multiple threads recording failures/successes simultaneously
6. configure() method — setting custom thresholds
"""

import threading
import time
from unittest.mock import patch

import pytest

from agent.circuit_breaker import ProviderCircuitBreaker, CircuitState, ProviderState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_breaker():
    """Reset the singleton between tests so state doesn't leak."""
    ProviderCircuitBreaker._instance = None
    cb = ProviderCircuitBreaker.get_instance()
    yield cb
    ProviderCircuitBreaker._instance = None


PROVIDER = "anthropic"


# ---------------------------------------------------------------------------
# 1. Threshold tripping
# ---------------------------------------------------------------------------

class TestThresholdTripping:
    """Breaker opens after N consecutive failures."""

    def test_stays_closed_below_threshold(self, fresh_breaker):
        cb = fresh_breaker
        cb.record_failure(PROVIDER, status_code=429)
        # Default threshold is 2; one failure should NOT trip the breaker
        assert cb.should_use_fallback(PROVIDER) is False
        status = cb.get_status()
        assert status[PROVIDER]["state"] == "closed"
        assert status[PROVIDER]["failures"] == 1

    def test_opens_at_threshold(self, fresh_breaker):
        cb = fresh_breaker
        for _ in range(cb.failure_threshold):
            cb.record_failure(PROVIDER, status_code=429)

        assert cb.should_use_fallback(PROVIDER) is True
        status = cb.get_status()
        assert status[PROVIDER]["state"] == "open"
        assert status[PROVIDER]["failures"] == cb.failure_threshold

    def test_opens_above_threshold(self, fresh_breaker):
        cb = fresh_breaker
        for _ in range(cb.failure_threshold + 5):
            cb.record_failure(PROVIDER, status_code=429)

        assert cb.should_use_fallback(PROVIDER) is True
        assert cb.get_status()[PROVIDER]["state"] == "open"

    def test_different_providers_independent(self, fresh_breaker):
        cb = fresh_breaker
        # Trip anthropic
        for _ in range(cb.failure_threshold):
            cb.record_failure("anthropic", status_code=429)
        # openai should still be closed
        assert cb.should_use_fallback("anthropic") is True
        assert cb.should_use_fallback("openai") is False

    def test_default_threshold_is_two(self, fresh_breaker):
        cb = fresh_breaker
        assert cb.failure_threshold == 2


# ---------------------------------------------------------------------------
# 2. Retry-After timeout behavior
# ---------------------------------------------------------------------------

class TestRetryAfterBehavior:
    """Breaker respects retry-after headers."""

    def test_retry_after_stored_on_failure(self, fresh_breaker):
        cb = fresh_breaker
        cb.record_failure(PROVIDER, status_code=429, retry_after=120)
        # Access internal state
        ps = cb._providers[PROVIDER]
        assert ps.last_429_retry_after == 120

    def test_retry_after_none_when_not_provided(self, fresh_breaker):
        cb = fresh_breaker
        cb.record_failure(PROVIDER, status_code=429)
        ps = cb._providers[PROVIDER]
        assert ps.last_429_retry_after is None

    def test_retry_after_updates_on_subsequent_failure(self, fresh_breaker):
        cb = fresh_breaker
        cb.record_failure(PROVIDER, status_code=429, retry_after=60)
        cb.record_failure(PROVIDER, status_code=429, retry_after=300)
        ps = cb._providers[PROVIDER]
        assert ps.last_429_retry_after == 300

    def test_retry_after_zero_does_not_store(self, fresh_breaker):
        """retry_after=0 is falsy and should not be stored."""
        cb = fresh_breaker
        cb.record_failure(PROVIDER, status_code=429, retry_after=0)
        ps = cb._providers[PROVIDER]
        assert ps.last_429_retry_after is None

    def test_large_retry_after_still_recorded(self, fresh_breaker):
        cb = fresh_breaker
        cb.record_failure(PROVIDER, status_code=429, retry_after=3600)
        cb.record_failure(PROVIDER, status_code=429, retry_after=3600)
        ps = cb._providers[PROVIDER]
        assert ps.last_429_retry_after == 3600
        assert ps.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# 3. Half-open probe selection
# ---------------------------------------------------------------------------

class TestHalfOpenProbe:
    """After reset_timeout, breaker enters half_open and allows one probe."""

    def _trip_breaker(self, cb, provider=PROVIDER):
        """Helper to trip the breaker open."""
        for _ in range(cb.failure_threshold):
            cb.record_failure(provider, status_code=429)

    def test_transitions_to_half_open_after_timeout(self, fresh_breaker):
        cb = fresh_breaker
        cb.reset_timeout = 1  # 1 second for fast testing
        self._trip_breaker(cb)
        assert cb.should_use_fallback(PROVIDER) is True

        time.sleep(1.1)

        # Next check should transition to half-open and allow the probe
        result = cb.should_use_fallback(PROVIDER, session_key="probe-session")
        assert result is False  # probe is allowed through
        assert cb._providers[PROVIDER].state == CircuitState.HALF_OPEN

    def test_probe_session_allowed_others_blocked(self, fresh_breaker):
        cb = fresh_breaker
        cb.reset_timeout = 0  # immediate transition for testing
        self._trip_breaker(cb)

        # First caller becomes the probe session
        assert cb.should_use_fallback(PROVIDER, session_key="probe-session") is False
        assert cb._providers[PROVIDER].state == CircuitState.HALF_OPEN
        assert cb._providers[PROVIDER].half_open_probe_session == "probe-session"

        # Other sessions are still blocked
        assert cb.should_use_fallback(PROVIDER, session_key="other-session") is True

        # Probe session is still allowed
        assert cb.should_use_fallback(PROVIDER, session_key="probe-session") is False

    def test_half_open_failure_returns_to_open(self, fresh_breaker):
        cb = fresh_breaker
        cb.reset_timeout = 0
        self._trip_breaker(cb)

        # Transition to half-open
        cb.should_use_fallback(PROVIDER, session_key="probe-session")
        assert cb._providers[PROVIDER].state == CircuitState.HALF_OPEN

        # Probe fails
        cb.record_failure(PROVIDER, status_code=429)
        assert cb._providers[PROVIDER].state == CircuitState.OPEN
        assert cb._providers[PROVIDER].half_open_probe_session is None

    def test_half_open_success_closes_circuit(self, fresh_breaker):
        cb = fresh_breaker
        cb.reset_timeout = 0
        self._trip_breaker(cb)

        # Transition to half-open
        cb.should_use_fallback(PROVIDER, session_key="probe-session")
        assert cb._providers[PROVIDER].state == CircuitState.HALF_OPEN

        # Probe succeeds
        cb.record_success(PROVIDER)
        assert cb._providers[PROVIDER].state == CircuitState.CLOSED
        assert cb._providers[PROVIDER].failure_count == 0

    def test_probe_session_cleared_on_success(self, fresh_breaker):
        cb = fresh_breaker
        cb.reset_timeout = 0
        self._trip_breaker(cb)

        cb.should_use_fallback(PROVIDER, session_key="probe-session")
        cb.record_success(PROVIDER)
        assert cb._providers[PROVIDER].half_open_probe_session is None

    def test_half_open_empty_session_key(self, fresh_breaker):
        """Default empty session key works for probe selection."""
        cb = fresh_breaker
        cb.reset_timeout = 0
        self._trip_breaker(cb)

        # Probe with empty session key — implementation uses a sentinel for anonymous probes
        result = cb.should_use_fallback(PROVIDER, session_key="")
        assert result is False
        assert cb._providers[PROVIDER].half_open_probe_session is not None


# ---------------------------------------------------------------------------
# 4. Success resets
# ---------------------------------------------------------------------------

class TestSuccessResets:
    """Successful calls close the circuit and reset all state."""

    def test_success_resets_closed_circuit(self, fresh_breaker):
        cb = fresh_breaker
        cb.record_failure(PROVIDER, status_code=500)
        assert cb._providers[PROVIDER].failure_count == 1
        cb.record_success(PROVIDER)
        ps = cb._providers[PROVIDER]
        assert ps.state == CircuitState.CLOSED
        assert ps.failure_count == 0
        assert ps.last_failure_at == 0.0

    def test_success_closes_open_circuit(self, fresh_breaker):
        cb = fresh_breaker
        for _ in range(cb.failure_threshold):
            cb.record_failure(PROVIDER, status_code=429)
        assert cb._providers[PROVIDER].state == CircuitState.OPEN

        cb.record_success(PROVIDER)
        ps = cb._providers[PROVIDER]
        assert ps.state == CircuitState.CLOSED
        assert ps.failure_count == 0
        assert ps.opened_at == 0.0
        assert ps.last_429_retry_after is None

    def test_success_on_unknown_provider_creates_closed(self, fresh_breaker):
        cb = fresh_breaker
        cb.record_success("new-provider")
        ps = cb._providers["new-provider"]
        assert ps.state == CircuitState.CLOSED
        assert ps.failure_count == 0

    def test_success_clears_retry_after(self, fresh_breaker):
        cb = fresh_breaker
        cb.record_failure(PROVIDER, status_code=429, retry_after=600)
        cb.record_failure(PROVIDER, status_code=429, retry_after=600)
        assert cb._providers[PROVIDER].last_429_retry_after == 600

        cb.record_success(PROVIDER)
        assert cb._providers[PROVIDER].last_429_retry_after is None

    def test_should_use_fallback_false_after_success(self, fresh_breaker):
        cb = fresh_breaker
        for _ in range(cb.failure_threshold):
            cb.record_failure(PROVIDER, status_code=429)
        assert cb.should_use_fallback(PROVIDER) is True

        cb.record_success(PROVIDER)
        assert cb.should_use_fallback(PROVIDER) is False


# ---------------------------------------------------------------------------
# 5. Concurrent access patterns
# ---------------------------------------------------------------------------

class TestConcurrentAccess:
    """Multiple threads recording failures/successes simultaneously."""

    def test_concurrent_failures_trip_breaker(self, fresh_breaker):
        cb = fresh_breaker
        cb.configure(failure_threshold=10)
        barrier = threading.Barrier(10)
        errors = []

        def record_failures():
            try:
                barrier.wait(timeout=5)
                cb.record_failure(PROVIDER, status_code=429)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_failures) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        ps = cb._providers[PROVIDER]
        assert ps.failure_count == 10
        assert ps.state == CircuitState.OPEN

    def test_concurrent_success_and_failure(self, fresh_breaker):
        cb = fresh_breaker
        cb.configure(failure_threshold=50)
        barrier = threading.Barrier(20)
        errors = []

        def record_failure_task():
            try:
                barrier.wait(timeout=5)
                cb.record_failure(PROVIDER, status_code=429)
            except Exception as e:
                errors.append(e)

        def record_success_task():
            try:
                barrier.wait(timeout=5)
                cb.record_success(PROVIDER)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(20):
            if i % 2 == 0:
                threads.append(threading.Thread(target=record_failure_task))
            else:
                threads.append(threading.Thread(target=record_success_task))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        # State should be consistent — either closed or open, no corruption
        ps = cb._providers[PROVIDER]
        assert ps.state in (CircuitState.CLOSED, CircuitState.OPEN)

    def test_concurrent_should_use_fallback(self, fresh_breaker):
        cb = fresh_breaker
        # Trip the breaker first
        for _ in range(cb.failure_threshold):
            cb.record_failure(PROVIDER, status_code=429)

        results = []
        barrier = threading.Barrier(10)
        errors = []

        def check_fallback(session):
            try:
                barrier.wait(timeout=5)
                result = cb.should_use_fallback(PROVIDER, session_key=session)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=check_fallback, args=(f"session-{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(results) == 10

    def test_no_deadlocks_under_contention(self, fresh_breaker):
        """Rapid mixed operations must not deadlock."""
        cb = fresh_breaker
        cb.configure(failure_threshold=5)
        barrier = threading.Barrier(30)
        errors = []

        def mixed_ops(thread_id):
            try:
                barrier.wait(timeout=5)
                for _ in range(100):
                    cb.should_use_fallback(PROVIDER, session_key=f"s-{thread_id}")
                    cb.record_failure(PROVIDER, status_code=429)
                    cb.record_success(PROVIDER)
                    cb.get_status()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_ops, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors


# ---------------------------------------------------------------------------
# 6. configure() method
# ---------------------------------------------------------------------------

class TestConfigure:
    """Setting custom thresholds via configure()."""

    def test_configure_failure_threshold(self, fresh_breaker):
        cb = fresh_breaker
        cb.configure(failure_threshold=5)
        assert cb.failure_threshold == 5

        # Should not trip after 4 failures
        for _ in range(4):
            cb.record_failure(PROVIDER, status_code=429)
        assert cb.should_use_fallback(PROVIDER) is False

        # Trips on 5th
        cb.record_failure(PROVIDER, status_code=429)
        assert cb.should_use_fallback(PROVIDER) is True

    def test_configure_reset_timeout(self, fresh_breaker):
        cb = fresh_breaker
        cb.configure(reset_timeout=1)
        assert cb.reset_timeout == 1

        for _ in range(cb.failure_threshold):
            cb.record_failure(PROVIDER, status_code=429)
        assert cb.should_use_fallback(PROVIDER) is True

        time.sleep(1.1)
        # Should now transition to half-open
        assert cb.should_use_fallback(PROVIDER, session_key="probe") is False

    def test_configure_partial_update(self, fresh_breaker):
        cb = fresh_breaker
        original_timeout = cb.reset_timeout
        cb.configure(failure_threshold=10)
        assert cb.failure_threshold == 10
        assert cb.reset_timeout == original_timeout  # unchanged

    def test_configure_none_values_ignored(self, fresh_breaker):
        cb = fresh_breaker
        original_threshold = cb.failure_threshold
        original_timeout = cb.reset_timeout
        cb.configure(failure_threshold=None, reset_timeout=None)
        assert cb.failure_threshold == original_threshold
        assert cb.reset_timeout == original_timeout

    def test_configure_both_values(self, fresh_breaker):
        cb = fresh_breaker
        cb.configure(failure_threshold=10, reset_timeout=600)
        assert cb.failure_threshold == 10
        assert cb.reset_timeout == 600


# ---------------------------------------------------------------------------
# 7. Singleton behavior
# ---------------------------------------------------------------------------

class TestSingleton:
    """get_instance() returns the same object."""

    def test_get_instance_returns_same_object(self, fresh_breaker):
        a = ProviderCircuitBreaker.get_instance()
        b = ProviderCircuitBreaker.get_instance()
        assert a is b

    def test_singleton_shares_state(self, fresh_breaker):
        a = ProviderCircuitBreaker.get_instance()
        a.record_failure(PROVIDER, status_code=429)

        b = ProviderCircuitBreaker.get_instance()
        assert b._providers[PROVIDER].failure_count == 1


# ---------------------------------------------------------------------------
# 8. get_status() output
# ---------------------------------------------------------------------------

class TestGetStatus:
    """get_status() returns well-formed status dicts."""

    def test_empty_status(self, fresh_breaker):
        cb = fresh_breaker
        assert cb.get_status() == {}

    def test_closed_status(self, fresh_breaker):
        cb = fresh_breaker
        cb.record_failure(PROVIDER, status_code=500)
        status = cb.get_status()
        assert PROVIDER in status
        info = status[PROVIDER]
        assert info["state"] == "closed"
        assert info["failures"] == 1
        assert info["open_since"] is None
        assert info["reset_in"] is None

    def test_open_status(self, fresh_breaker):
        cb = fresh_breaker
        for _ in range(cb.failure_threshold):
            cb.record_failure(PROVIDER, status_code=429)
        status = cb.get_status()
        info = status[PROVIDER]
        assert info["state"] == "open"
        assert info["failures"] == cb.failure_threshold
        assert info["open_since"] is not None
        assert info["reset_in"] is not None
        assert info["reset_in"] > 0

    def test_multiple_providers_status(self, fresh_breaker):
        cb = fresh_breaker
        cb.record_failure("anthropic", status_code=429)
        cb.record_failure("openai", status_code=429)
        status = cb.get_status()
        assert "anthropic" in status
        assert "openai" in status

    def test_status_after_success_reset(self, fresh_breaker):
        cb = fresh_breaker
        for _ in range(cb.failure_threshold):
            cb.record_failure(PROVIDER, status_code=429)
        cb.record_success(PROVIDER)
        status = cb.get_status()
        info = status[PROVIDER]
        assert info["state"] == "closed"
        assert info["failures"] == 0


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_record_failure_without_status_code(self, fresh_breaker):
        cb = fresh_breaker
        cb.record_failure(PROVIDER)  # no status_code, no retry_after
        assert cb._providers[PROVIDER].failure_count == 1

    def test_closed_provider_returns_false(self, fresh_breaker):
        cb = fresh_breaker
        # Never touched provider
        assert cb.should_use_fallback("brand-new-provider") is False

    def test_failure_count_persists_across_multiple_status_codes(self, fresh_breaker):
        cb = fresh_breaker
        cb.record_failure(PROVIDER, status_code=429)
        cb.record_failure(PROVIDER, status_code=529)
        assert cb._providers[PROVIDER].failure_count == 2
        assert cb._providers[PROVIDER].state == CircuitState.OPEN

    def test_last_failure_at_updated(self, fresh_breaker):
        cb = fresh_breaker
        before = time.time()
        cb.record_failure(PROVIDER, status_code=429)
        after = time.time()
        assert before <= cb._providers[PROVIDER].last_failure_at <= after

    def test_opened_at_set_when_tripped(self, fresh_breaker):
        cb = fresh_breaker
        before = time.time()
        for _ in range(cb.failure_threshold):
            cb.record_failure(PROVIDER, status_code=429)
        after = time.time()
        assert before <= cb._providers[PROVIDER].opened_at <= after

    def test_threshold_of_one(self, fresh_breaker):
        cb = fresh_breaker
        cb.configure(failure_threshold=1)
        cb.record_failure(PROVIDER, status_code=429)
        assert cb.should_use_fallback(PROVIDER) is True
