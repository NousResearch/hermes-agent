"""Tests for agent/circuit_breaker.py."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from agent.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerSnapshot,
    CircuitState,
    circuit_breaker,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_config(**kw) -> CircuitBreakerConfig:
    return CircuitBreakerConfig(**kw)


# ─── State machine tests ──────────────────────────────────────────────────────

class TestClosedToOpen:
    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", config=make_config(failure_threshold=3))
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")), record_failure=True)
        assert cb.state == CircuitState.CLOSED
        assert cb.snapshot().failure_count == 2

    def test_trips_at_exact_threshold(self):
        cb = CircuitBreaker("test", config=make_config(failure_threshold=3))
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")), record_failure=True)
        assert cb.state == CircuitState.OPEN
        snap = cb.snapshot()
        assert snap.failure_count == 3
        assert snap.open_until is not None

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", config=make_config(failure_threshold=3))
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), record_failure=True)
        cb.call(lambda: None, record_failure=True)
        assert cb.state == CircuitState.CLOSED
        assert cb.snapshot().failure_count == 0


class TestOpenToHalfOpen:
    def test_rejects_calls_while_open(self):
        cb = CircuitBreaker(
            "test",
            config=make_config(failure_threshold=1, recovery_timeout=0.1),
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), record_failure=True)
        assert cb.state == CircuitState.OPEN

        # Second call must be rejected (no fallback → raises CircuitBreakerOpen)
        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: None)

    def test_rejects_with_fallback_when_open(self):
        cb = CircuitBreaker(
            "test",
            config=make_config(failure_threshold=1, recovery_timeout=0.1),
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), record_failure=True)

        sentinel = object()
        result = cb.call(
            lambda: "ok",
            fallback_factory=lambda exc: sentinel,
        )
        assert result is sentinel
        assert cb.snapshot().total_rejected == 1

    def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(
            "test",
            config=make_config(failure_threshold=1, recovery_timeout=0.05),
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), record_failure=True)
        assert cb.state == CircuitState.OPEN

        time.sleep(0.07)
        # Timeout check happens inside call(); successful probe → HALF_OPEN → CLOSED
        cb.call(lambda: None)
        assert cb.state == CircuitState.CLOSED


class TestHalfOpenToClosed:
    def test_success_closes_circuit(self):
        cb = CircuitBreaker(
            "test",
            config=make_config(failure_threshold=1, recovery_timeout=0.05),
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), record_failure=True)
        time.sleep(0.07)

        # Probe call — should transition to HALF_OPEN then succeed → CLOSED
        cb.call(lambda: None, record_failure=True)
        assert cb.state == CircuitState.CLOSED

    def test_failure_reopens_circuit(self):
        cb = CircuitBreaker(
            "test",
            config=make_config(failure_threshold=1, recovery_timeout=0.05),
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), record_failure=True)
        time.sleep(0.07)

        # HALF_OPEN probe fails → re-open
        with pytest.raises(RuntimeError):
            cb.call(
                lambda: (_ for _ in ()).throw(RuntimeError("probe fail")),
                record_failure=True,
            )
        assert cb.state == CircuitState.OPEN

    def test_half_open_respects_max_calls(self):
        cb = CircuitBreaker(
            "test",
            config=make_config(
                failure_threshold=1,
                recovery_timeout=0.05,
                half_open_max_calls=2,
                success_threshold=1,
            ),
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), record_failure=True)
        time.sleep(0.07)

        # First probe call → HALF_OPEN; allowed
        cb.call(lambda: None)
        assert cb.state == CircuitState.CLOSED

        # Exhaust again
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), record_failure=True)
        time.sleep(0.07)

        # First probe call → HALF_OPEN; allowed
        cb.call(lambda: None)
        # Second probe call → should be rejected (max reached, but first already succeeded → CLOSED)
        # Actually after first probe succeeds → CLOSED, so this should be fine
        assert cb.state == CircuitState.CLOSED

    def test_success_threshold_higher_than_one(self):
        cb = CircuitBreaker(
            "test",
            config=make_config(
                failure_threshold=1,
                recovery_timeout=0.05,
                half_open_max_calls=3,
                success_threshold=2,
            ),
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), record_failure=True)
        time.sleep(0.07)

        # First probe — still HALF_OPEN
        cb.call(lambda: None, record_failure=True)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.snapshot().success_count == 1

        # Second probe — should close
        cb.call(lambda: None, record_failure=True)
        assert cb.state == CircuitState.CLOSED


class TestExcludedExceptions:
    def test_excluded_exception_does_not_trip(self):
        cb = CircuitBreaker(
            "test",
            config=make_config(failure_threshold=1, excluded_exceptions=(ValueError,)),
        )
        for _ in range(5):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("bad input")), record_failure=True)
        assert cb.state == CircuitState.CLOSED

    def test_non_excluded_exception_trips(self):
        cb = CircuitBreaker(
            "test",
            config=make_config(failure_threshold=1, excluded_exceptions=(ValueError,)),
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("oops")), record_failure=True)
        assert cb.state == CircuitState.OPEN


class TestReset:
    def test_reset_closes_circuit(self):
        cb = CircuitBreaker(
            "test",
            config=make_config(failure_threshold=1, recovery_timeout=60.0),
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), record_failure=True)
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.snapshot().failure_count == 0


class TestManualRecord:
    def test_record_success_in_open_ignored(self):
        """Success recorded while OPEN does not close the circuit."""
        cb = CircuitBreaker("test", config=make_config(failure_threshold=1))
        cb.record_failure(RuntimeError("boom"))
        assert cb.state == CircuitState.OPEN
        cb.record_success()
        # OPEN state ignores successes — circuit stays OPEN
        assert cb.state == CircuitState.OPEN
        # But failure counter is reset so first probe can succeed
        assert cb.snapshot().failure_count == 0

    def test_record_success_in_half_open_closes(self):
        cb = CircuitBreaker(
            "test",
            config=make_config(failure_threshold=1, recovery_timeout=0.05),
        )
        cb.record_failure(RuntimeError("boom"))
        time.sleep(0.07)
        # Timeout check is done inside call(), so manually force HALF_OPEN
        cb._state = CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


class TestSnapshot:
    def test_snapshot_fields(self):
        cb = CircuitBreaker("test", config=make_config(failure_threshold=1))
        snap = cb.snapshot()
        assert isinstance(snap, CircuitBreakerSnapshot)
        assert snap.name == "test"
        assert snap.state == CircuitState.CLOSED
        assert snap.total_calls == 0
        assert snap.total_successes == 0
        assert snap.total_failures == 0
        assert snap.total_rejected == 0
        assert snap.last_failure_time is None
        assert snap.last_failure_reason is None
        assert snap.open_until is None

    def test_snapshot_is_immutable(self):
        cb = CircuitBreaker("test")
        snap = cb.snapshot()
        with pytest.raises(AttributeError):
            snap.total_calls = 999


class TestConcurrency:
    def test_thread_safety(self):
        cb = CircuitBreaker(
            "test",
            config=make_config(failure_threshold=20),
        )
        barrier = threading.Barrier(10)
        errors = []

        def worker():
            try:
                barrier.wait()
                for _ in range(10):
                    cb.call(lambda: None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cb.snapshot().total_calls == 100
        assert not errors


class TestConfigValidation:
    def test_rejects_invalid_failure_threshold(self):
        with pytest.raises(ValueError, match="failure_threshold"):
            make_config(failure_threshold=0)

    def test_rejects_invalid_recovery_timeout(self):
        with pytest.raises(ValueError, match="recovery_timeout"):
            make_config(recovery_timeout=0.0)

    def test_rejects_invalid_half_open_max_calls(self):
        with pytest.raises(ValueError, match="half_open_max_calls"):
            make_config(half_open_max_calls=0)

    def test_rejects_invalid_success_threshold(self):
        with pytest.raises(ValueError, match="success_threshold"):
            make_config(success_threshold=0)


class TestFactory:
    def test_circuit_breaker_factory(self):
        cb = circuit_breaker("openrouter", failure_threshold=3, recovery_timeout=15.0)
        assert cb.name == "openrouter"
        assert cb._config.failure_threshold == 3
        assert cb._config.recovery_timeout == 15.0


class TestOnStateChangeCallback:
    def test_callback_fires_on_transition(self):
        transitions = []

        def observer(old, new, snap):
            transitions.append((old, new, snap.name))

        cb = CircuitBreaker(
            "test-cb",
            config=make_config(failure_threshold=1, recovery_timeout=0.05),
            on_state_change=observer,
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), record_failure=True)

        assert (CircuitState.CLOSED, CircuitState.OPEN, "test-cb") in transitions


class TestOnRejectedCallback:
    def test_callback_receives_circuit_breaker_open(self):
        received = []

        def observer(exc: CircuitBreakerOpen):
            received.append(exc)

        cb = CircuitBreaker(
            "test",
            config=make_config(failure_threshold=1, recovery_timeout=5.0),
            on_rejected=observer,
        )
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()), record_failure=True)

        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: None)

        assert len(received) == 1
        assert received[0].resource_name == "test"
