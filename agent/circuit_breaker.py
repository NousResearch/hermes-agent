"""Process-wide circuit breaker for provider rate limits.

This prevents multiple concurrent agents from hammering the same provider after
that provider has already started returning 429/529-style overload errors.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ProviderState:
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_at: float = 0.0
    opened_at: float = 0.0
    open_timeout: int = 0
    last_429_retry_after: Optional[int] = None
    half_open_probe_session: Optional[str] = None


class ProviderCircuitBreaker:
    """Singleton circuit breaker shared across all agents in a gateway process.

    CLOSED:
        Normal operation. Failures increment a counter.
    OPEN:
        Provider is considered unhealthy/rate-limited. Requests SHOULD use
        fallback immediately until the reset timeout elapses.
    HALF_OPEN:
        One designated session is allowed to probe the provider. Success closes
        the circuit, failure re-opens it.
    """

    _instance: Optional["ProviderCircuitBreaker"] = None
    _instance_lock = threading.Lock()

    DEFAULT_FAILURE_THRESHOLD = 2
    DEFAULT_RESET_TIMEOUT = 300
    DEFAULT_HALF_OPEN_TIMEOUT = 30

    @classmethod
    def get_instance(cls) -> "ProviderCircuitBreaker":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._providers: Dict[str, ProviderState] = {}
        self.failure_threshold = self.DEFAULT_FAILURE_THRESHOLD
        self.reset_timeout = self.DEFAULT_RESET_TIMEOUT
        self.half_open_timeout = self.DEFAULT_HALF_OPEN_TIMEOUT

    def _get_state(self, provider: str) -> ProviderState:
        key = (provider or "unknown").strip().lower() or "unknown"
        if key not in self._providers:
            self._providers[key] = ProviderState()
        return self._providers[key]

    def should_use_fallback(self, provider: str, session_key: str = "") -> bool:
        """Return True if the provider circuit is open for this session."""
        with self._lock:
            state = self._get_state(provider)
            now = time.time()

            if state.state == CircuitState.CLOSED:
                return False

            if state.state == CircuitState.OPEN:
                elapsed = now - state.opened_at
                timeout = state.open_timeout or self.reset_timeout
                if elapsed >= timeout:
                    state.state = CircuitState.HALF_OPEN
                    state.half_open_probe_session = session_key or "__anonymous_probe__"
                    logger.info(
                        "Circuit breaker %s: OPEN -> HALF_OPEN (probe=%s)",
                        provider,
                        state.half_open_probe_session[:32],
                    )
                    return False
                return True

            if state.state == CircuitState.HALF_OPEN:
                probe_session = state.half_open_probe_session or "__anonymous_probe__"
                current_session = session_key or "__anonymous_probe__"
                return probe_session != current_session

            return False

    def record_failure(
        self,
        provider: str,
        status_code: Optional[int] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        """Record a retry-worthy provider failure.

        Intended for 429, 529, and overload-style failures that SHOULD trigger
        cross-session provider suppression.
        """
        with self._lock:
            state = self._get_state(provider)
            now = time.time()
            state.failure_count += 1
            state.last_failure_at = now
            if retry_after and retry_after > 0:
                state.last_429_retry_after = retry_after

            next_timeout = max(self.reset_timeout, retry_after or 0)

            if state.state == CircuitState.HALF_OPEN:
                state.state = CircuitState.OPEN
                state.opened_at = now
                state.open_timeout = next_timeout
                state.half_open_probe_session = None
                logger.warning(
                    "Circuit breaker %s: HALF_OPEN -> OPEN after failed probe (timeout=%ss, status=%s)",
                    provider,
                    next_timeout,
                    status_code,
                )
                return

            if state.failure_count >= self.failure_threshold:
                if state.state != CircuitState.OPEN:
                    logger.warning(
                        "Circuit breaker %s: CLOSED -> OPEN after %d failures (timeout=%ss, status=%s)",
                        provider,
                        state.failure_count,
                        next_timeout,
                        status_code,
                    )
                state.state = CircuitState.OPEN
                state.opened_at = now
                state.open_timeout = next_timeout
                state.half_open_probe_session = None

    def record_success(self, provider: str) -> None:
        """Record a successful request and fully close the circuit."""
        with self._lock:
            state = self._get_state(provider)
            previous = state.state
            state.state = CircuitState.CLOSED
            state.failure_count = 0
            state.last_failure_at = 0.0
            state.opened_at = 0.0
            state.open_timeout = 0
            state.last_429_retry_after = None
            state.half_open_probe_session = None
            if previous != CircuitState.CLOSED:
                logger.info("Circuit breaker %s: %s -> CLOSED", provider, previous.value)

    def get_status(self) -> Dict[str, dict]:
        """Return human-readable circuit breaker state for all providers."""
        with self._lock:
            now = time.time()
            result: Dict[str, dict] = {}
            for provider, state in self._providers.items():
                reset_in = None
                if state.state == CircuitState.OPEN:
                    timeout = state.open_timeout or self.reset_timeout
                    reset_in = max(0, timeout - (now - state.opened_at))
                result[provider] = {
                    "state": state.state.value,
                    "failures": state.failure_count,
                    "last_failure": state.last_failure_at or None,
                    "open_since": state.opened_at or None,
                    "reset_in": reset_in,
                    "retry_after": state.last_429_retry_after,
                    "probe_session": state.half_open_probe_session,
                }
            return result

    def configure(
        self,
        failure_threshold: Optional[int] = None,
        reset_timeout: Optional[int] = None,
        half_open_timeout: Optional[int] = None,
    ) -> None:
        """Update thresholds from config."""
        with self._lock:
            if failure_threshold is not None and failure_threshold > 0:
                self.failure_threshold = int(failure_threshold)
            if reset_timeout is not None and reset_timeout > 0:
                self.reset_timeout = int(reset_timeout)
            if half_open_timeout is not None and half_open_timeout > 0:
                self.half_open_timeout = int(half_open_timeout)
