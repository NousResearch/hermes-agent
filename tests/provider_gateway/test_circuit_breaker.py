import threading
import time
import pytest
from provider_gateway.circuit_breaker import (
    BreakerConfig,
    CircuitBreaker,
    CircuitState,
)


def test_circuit_breaker_default_closed() -> None:
    """Test that a new circuit breaker is CLOSED and available."""
    breaker = CircuitBreaker()
    assert breaker.is_available("anthropic") is True
    health = breaker.get_health("anthropic")
    assert health is None  # Lazy creation


def test_circuit_breaker_trips_to_open() -> None:
    """Test that circuit trips to OPEN after failure threshold is reached."""
    config = BreakerConfig(failure_threshold=3, reset_timeout_ms=50)
    breaker = CircuitBreaker(config)

    # First two failures
    breaker.record_failure("anthropic")
    breaker.record_failure("anthropic")
    assert breaker.is_available("anthropic") is True
    assert breaker.get_health("anthropic").state == CircuitState.CLOSED
    assert breaker.get_health("anthropic").consecutive_failures == 2

    # Third failure trips
    breaker.record_failure("anthropic")
    assert breaker.is_available("anthropic") is False
    assert breaker.get_health("anthropic").state == CircuitState.OPEN
    assert breaker.get_health("anthropic").consecutive_failures == 3


def test_circuit_breaker_cooldown_to_half_open_and_recovery() -> None:
    """Test cooldown timeout transition to HALF_OPEN and back to CLOSED on success."""
    config = BreakerConfig(failure_threshold=2, reset_timeout_ms=20)
    breaker = CircuitBreaker(config)

    breaker.record_failure("anthropic")
    breaker.record_failure("anthropic")
    assert breaker.is_available("anthropic") is False

    # Wait for cooldown to expire
    time.sleep(0.025)

    # Accessing is_available triggers transition to HALF_OPEN
    assert breaker.is_available("anthropic") is True
    assert breaker.get_health("anthropic").state == CircuitState.HALF_OPEN

    # Record success recovers to CLOSED
    breaker.record_success("anthropic", latency_ms=150.0)
    assert breaker.get_health("anthropic").state == CircuitState.CLOSED
    assert breaker.get_health("anthropic").consecutive_failures == 0
    assert breaker.is_available("anthropic") is True


def test_circuit_breaker_half_open_failure_re_trips() -> None:
    """Test that failure in HALF_OPEN immediately trips back to OPEN with backoff."""
    config = BreakerConfig(failure_threshold=2, reset_timeout_ms=20)
    breaker = CircuitBreaker(config)

    breaker.record_failure("anthropic")
    breaker.record_failure("anthropic")
    assert breaker.is_available("anthropic") is False

    time.sleep(0.025)
    assert breaker.is_available("anthropic") is True  # transitioned to HALF_OPEN

    # Failure in HALF_OPEN trips immediately to OPEN
    breaker.record_failure("anthropic")
    assert breaker.is_available("anthropic") is False
    health = breaker.get_health("anthropic")
    assert health.state == CircuitState.OPEN
    assert health.backoff_level == 1  # Incremented backoff level


def test_latency_p50_calculation() -> None:
    """Test median P50 latency calculations on success records."""
    breaker = CircuitBreaker()
    provider = "openrouter"

    breaker.record_success(provider, 100.0)
    breaker.record_success(provider, 200.0)
    breaker.record_success(provider, 300.0)
    health = breaker.get_health(provider)
    assert health.latency_p50 == 200.0

    # Even number of elements
    breaker.record_success(provider, 400.0)
    # sorted: 100, 200, 300, 400. Median = (200 + 300) / 2 = 250
    assert health.latency_p50 == 250.0

    # Test error rate
    breaker.record_failure(provider)
    # 4 success + 1 failure = 5 total
    assert health.error_rate == 0.20


def test_circuit_breaker_concurrency() -> None:
    """Test that recording success and failure works fine in concurrent environment."""
    breaker = CircuitBreaker()
    provider = "anthropic"

    def record_failures():
        for _ in range(50):
            breaker.record_failure(provider)

    def record_successes():
        for _ in range(50):
            breaker.record_success(provider, 50.0)

    threads = [
        threading.Thread(target=record_failures),
        threading.Thread(target=record_successes),
        threading.Thread(target=record_failures),
        threading.Thread(target=record_successes),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    health = breaker.get_health(provider)
    assert health.total_requests == 200
    assert health.total_failures == 100
