"""Circuit breaker for the opt-in provider gateway.

Monitors provider failure rates and latency to prevent routing to degraded
backends. Thread-safe implementation.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """The operational state of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class BreakerConfig:
    """Configuration options for a circuit breaker."""

    failure_threshold: int = 5
    reset_timeout_ms: int = 60000  # Time to wait in OPEN state before trying HALF_OPEN
    max_latency_samples: int = 50


class ProviderHealth:
    """Stateful health tracker for a single provider."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time: float | None = None
        self.last_success_time: float | None = None
        self.total_requests = 0
        self.total_failures = 0
        self.latency_samples: list[float] = []
        self.backoff_level = 0

    @property
    def latency_p50(self) -> float:
        """Return the median (P50) latency of successful requests in milliseconds."""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        n = len(sorted_samples)
        if n % 2 == 1:
            return sorted_samples[n // 2]
        return (sorted_samples[n // 2 - 1] + sorted_samples[n // 2]) / 2.0

    @property
    def error_rate(self) -> float:
        """Return the ratio of failed requests to total requests."""
        if self.total_requests == 0:
            return 0.0
        return self.total_failures / self.total_requests


class CircuitBreaker:
    """Thread-safe circuit breaker manager for multiple LLM providers."""

    def __init__(self, config: BreakerConfig | None = None) -> None:
        self.config = config if config is not None else BreakerConfig()
        self._providers: dict[str, ProviderHealth] = {}
        self._lock = threading.Lock()

    def is_available(self, provider: str) -> bool:
        """Return True if the provider is healthy enough to accept requests."""
        with self._lock:
            health = self._providers.get(provider)
            if health is None:
                return True

            if health.state == CircuitState.CLOSED:
                return True

            if health.state == CircuitState.OPEN:
                if health.last_failure_time is None:
                    health.state = CircuitState.CLOSED
                    return True

                elapsed_ms = (time.time() - health.last_failure_time) * 1000.0
                cooldown_ms = self.config.reset_timeout_ms * (2**health.backoff_level)
                if elapsed_ms >= cooldown_ms:
                    health.state = CircuitState.HALF_OPEN
                    logger.debug(
                        "Circuit breaker for provider %s transitioned from OPEN to HALF_OPEN",
                        provider,
                    )
                    return True
                return False

            if health.state == CircuitState.HALF_OPEN:
                return True

            return True

    def record_success(self, provider: str, latency_ms: float = 0.0) -> None:
        """Record a successful provider request and reset failures."""
        with self._lock:
            health = self._get_or_create(provider)
            health.total_requests += 1
            health.last_success_time = time.time()
            health.consecutive_failures = 0
            health.backoff_level = 0

            if latency_ms > 0:
                health.latency_samples.append(latency_ms)
                if len(health.latency_samples) > self.config.max_latency_samples:
                    health.latency_samples.pop(0)

            if health.state != CircuitState.CLOSED:
                logger.info(
                    "Circuit breaker for provider %s recovered. State changed to CLOSED.",
                    provider,
                )
                health.state = CircuitState.CLOSED

    def record_failure(self, provider: str) -> None:
        """Record a failed provider request and possibly trip the circuit."""
        with self._lock:
            health = self._get_or_create(provider)
            health.total_requests += 1
            health.total_failures += 1
            health.consecutive_failures += 1
            health.last_failure_time = time.time()

            if health.state == CircuitState.CLOSED:
                if health.consecutive_failures >= self.config.failure_threshold:
                    health.state = CircuitState.OPEN
                    logger.warning(
                        "Circuit breaker for provider %s TRIPPED to OPEN after %d consecutive failures",
                        provider,
                        health.consecutive_failures,
                    )
            elif health.state == CircuitState.HALF_OPEN:
                # Failure in HALF_OPEN trips immediately back to OPEN with increased backoff
                health.state = CircuitState.OPEN
                health.backoff_level = min(health.backoff_level + 1, 5)  # Cap backoff level at 5 (32x timeout)
                logger.warning(
                    "Circuit breaker for provider %s failed in HALF_OPEN. Tripped to OPEN with backoff level %d",
                    provider,
                    health.backoff_level,
                )

    def get_health(self, provider: str) -> ProviderHealth | None:
        """Return the health metrics for a given provider, if tracked."""
        with self._lock:
            return self._providers.get(provider)

    def get_all_health(self) -> dict[str, ProviderHealth]:
        """Return a snapshot copy of all tracked provider healths."""
        with self._lock:
            return dict(self._providers)

    def _get_or_create(self, provider: str) -> ProviderHealth:
        if provider not in self._providers:
            self._providers[provider] = ProviderHealth(provider)
        return self._providers[provider]
