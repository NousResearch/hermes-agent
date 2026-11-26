"""
Circuit Breaker — Cascading-failure prevention for Hermes Agent.

Implements the standard three-state machine (CLOSED / OPEN / HALF_OPEN)
to halt calls to a failing service and give it time to recover before
allowing probe requests through.

Design principles applied:
  GL (Generative Loop)      — self-healing: CLOSED→OPEN→HALF_OPEN→CLOSED
                              closes automatically when health is restored.
  OS (Observable State)    — all internal counters and state are readable
                              via snapshot(); no hidden variables.
  MI (Module Independence) — pure-Python, no external dependencies,
                              drop-in for any callable or provider.
  LP (Least Privilege)     — circuit trips only on the specific named
                              resource; other providers are unaffected.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    """Possible circuit breaker states."""
    CLOSED   = auto()   # Normal operation; requests pass through.
    OPEN     = auto()   # Failing; requests are blocked immediately.
    HALF_OPEN = auto()  # Recovery probe; one or more test requests allowed.


# ─── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class CircuitBreakerConfig:
    """
    Tunable thresholds for a single circuit breaker instance.

    Args:
        failure_threshold: Number of consecutive failures needed to trip
            the circuit from CLOSED → OPEN.
        recovery_timeout: Seconds to wait in OPEN state before transitioning
            to HALF_OPEN to test recovery.
        half_open_max_calls: How many test calls are allowed in HALF_OPEN
            before deciding success (default 1).
        success_threshold: Number of successes needed in HALF_OPEN to
            close the circuit (default 1).  Set > 1 to require multiple
            healthy responses before declaring recovery.
        excluded_exceptions: Exception types that increment the failure
            counter but are NOT eligible for circuit-breaker protection
            (e.g. ``ValueError`` for bad input — those should never trip
            the circuit).
    """
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 1
    success_threshold: int = 1
    excluded_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (ValueError, TypeError)
    )

    def __post_init__(self):
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be >= 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")


# ─── Snapshot ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CircuitBreakerSnapshot:
    """
    Immutable read-only view of circuit breaker state.
    Returned by ``CircuitBreaker.snapshot()`` so callers can inspect
    without acquiring the lock.
    """
    name: str
    state: CircuitState
    failure_count: int          # consecutive failures in CLOSED / half-open failures in HALF_OPEN
    success_count: int          # consecutive successes in HALF_OPEN only
    last_failure_time: Optional[float]   # time.monotonic() of most recent failure, or None
    last_failure_reason: Optional[str]   # exception repr or message
    open_until: Optional[float]          # monotonic deadline when OPEN expires; None if not OPEN
    total_calls: int
    total_successes: int
    total_failures: int
    total_rejected: int         # calls rejected while circuit was OPEN

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN


# ─── Core ─────────────────────────────────────────────────────────────────────

class CircuitBreakerOpen(RuntimeError):
    """
    Raised by ``CircuitBreaker.call()`` when the circuit is OPEN and the
    caller chose to propagate the blocked-state as an error rather than
    silently returning a fallback value.

    The ``from_original`` attribute carries the underlying exception when
    ``record_failure=True`` is used.
    """

    def __init__(self, resource_name: str, open_until: float):
        self.resource_name = resource_name
        self.open_until = open_until
        wait_secs = max(0.0, open_until - time.monotonic())
        super().__init__(
            f"Circuit breaker for '{resource_name}' is OPEN; "
            f"retry after {wait_secs:.1f}s (at t={open_until:.1f})"
        )
        self.from_original: Optional[Exception] = None


class CircuitBreaker:
    """
    Thread-safe circuit breaker with CLOSED / OPEN / HALF_OPEN state machine.

    Usage — basic::

        cb = CircuitBreaker(name="openrouter", config=CircuitBreakerConfig())
        cb.call(my_api_function, *args, **kwargs)

    Usage — with fallback::

        result = cb.call(
            my_api_function, *args,
            fallback_factory=lambda exc: default_value,
            record_failure=True,
        )

    State transitions::

        CLOSED ──(failure_threshold reached)──→ OPEN
        OPEN ────(recovery_timeout elapsed)────→ HALF_OPEN
        HALF_OPEN─(success_threshold reached)──→ CLOSED
        HALF_OPEN─(failure)──────────────────→ OPEN (reset timer)

    Observable metrics (via ``snapshot()``):
        - state, failure_count, success_count, total_calls,
          total_successes, total_failures, total_rejected

    Args:
        name: Human-readable identifier (used in log messages and exceptions).
        config: CircuitBreakerConfig thresholds.
        on_state_change: Optional callback ``(old_state, new_state, snapshot) -> None``.
        on_rejected: Optional callback ``(exc: CircuitBreakerOpen) -> None``.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState, CircuitBreakerSnapshot], None]] = None,
        on_rejected: Optional[Callable[[CircuitBreakerOpen], None]] = None,
    ):
        self.name = name
        self._config = config or CircuitBreakerConfig()
        self._on_state_change = on_state_change
        self._on_rejected = on_rejected

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_failure_reason: Optional[str] = None
        self._open_until: Optional[float] = None      # monotonic deadline when OPEN expires
        self._half_open_calls = 0                    # test calls made in HALF_OPEN

        # Accumulated counters (never reset — useful for dashboards)
        self._total_calls = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_rejected = 0

        self._lock = threading.RLock()

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        """Current circuit state (no lock needed for read of Enum)."""
        return self._state

    def snapshot(self) -> CircuitBreakerSnapshot:
        """Return an immutable snapshot of all counters and state."""
        with self._lock:
            return CircuitBreakerSnapshot(
                name=self.name,
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                last_failure_time=self._last_failure_time,
                last_failure_reason=self._last_failure_reason,
                open_until=self._open_until,
                total_calls=self._total_calls,
                total_successes=self._total_successes,
                total_failures=self._total_failures,
                total_rejected=self._total_rejected,
            )

    def call(
        self,
        func: Callable[..., _T],
        *args,
        record_failure: bool = False,
        fallback_factory: Optional[Callable[[Exception], _T]] = None,
        **kwargs,
    ) -> _T:
        """
        Execute ``func(*args, **kwargs)`` through the circuit breaker.

        Args:
            func: The callable to execute.
            record_failure: If True and the call raises an exception that is
                NOT in ``excluded_exceptions``, record it as a failure and
                potentially trip the circuit.
            fallback_factory: If provided and the circuit is OPEN, call
                ``fallback_factory(original_exc)`` and return its result
                instead of raising ``CircuitBreakerOpen``.
                If not provided and the circuit is OPEN, raises
                ``CircuitBreakerOpen``.

        Returns:
            The return value of ``func(*args, **kwargs)``, or the result of
            ``fallback_factory(original_exc)`` when the circuit is OPEN and
            a fallback is registered.

        Raises:
            CircuitBreakerOpen: When the circuit is OPEN and no fallback is
                registered.
        """
        with self._lock:
            self._total_calls += 1

            # Fast path: OPEN → check timeout
            if self._state == CircuitState.OPEN:
                if self._open_until is not None and time.monotonic() >= self._open_until:
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self._total_rejected += 1
                    exc = CircuitBreakerOpen(self.name, self._open_until or 0.0)
                    if self._on_rejected:
                        try:
                            self._on_rejected(exc)
                        except Exception:
                            pass
                    if fallback_factory is not None:
                        return fallback_factory(exc)
                    raise exc

            # HALF_OPEN: enforce max probe count
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._config.half_open_max_calls:
                    self._total_rejected += 1
                    exc = CircuitBreakerOpen(self.name, self._open_until or 0.0)
                    if fallback_factory is not None:
                        return fallback_factory(exc)
                    raise exc
                self._half_open_calls += 1

        # ── Execute outside the lock ────────────────────────────────────────────
        original_exc: Optional[Exception] = None
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as exc:
            original_exc = exc
            if record_failure and not self._is_excluded(exc):
                self._record_failure(exc)
            raise

    def record_success(self) -> None:
        """Manually record a success event (e.g. after a side-effect call)."""
        self._record_success()

    def record_failure(self, exc: Optional[Exception] = None) -> None:
        """Manually record a failure event (e.g. after a side-effect call)."""
        self._record_failure(exc)

    def reset(self) -> None:
        """
        Force the circuit to CLOSED and zero all counters.
        Use for testing or administrative recovery.
        """
        with self._lock:
            old = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._open_until = None
            self._last_failure_time = None
            self._last_failure_reason = None
            if old != CircuitState.CLOSED:
                self._emit_state_change(old, CircuitState.CLOSED)
            logger.info("Circuit breaker '%s' manually reset to CLOSED", self.name)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _is_excluded(self, exc: Exception) -> bool:
        return isinstance(exc, self._config.excluded_exceptions)

    def _record_success(self) -> None:
        with self._lock:
            self._total_successes += 1
            self._failure_count = 0
            self._last_failure_reason = None

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def _record_failure(self, exc: Optional[Exception] = None) -> None:
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            now = time.monotonic()
            self._last_failure_time = now
            self._last_failure_reason = repr(exc) if exc is not None else "unknown"

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._open_until = now + self._config.recovery_timeout
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        "Circuit breaker '%s' OPEN after %d consecutive failures "
                        "(last error: %s). Recovery probe in %.1fs.",
                        self.name,
                        self._failure_count,
                        self._last_failure_reason,
                        self._config.recovery_timeout,
                    )

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN immediately trips back to OPEN
                self._half_open_calls = 0
                self._success_count = 0
                self._open_until = now + self._config.recovery_timeout
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "Circuit breaker '%s' HALF_OPEN→OPEN (probe failed: %s). "
                    "Retrying in %.1fs.",
                    self.name,
                    self._last_failure_reason,
                    self._config.recovery_timeout,
                )

    def _transition_to(self, new_state: CircuitState) -> None:
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._open_until = None
            self._last_failure_time = None
            self._last_failure_reason = None
        elif new_state == CircuitState.OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        self._emit_state_change(old, new_state)

    def _emit_state_change(self, old: CircuitState, new: CircuitState) -> None:
        if self._on_state_change is not None:
            try:
                snap = self.snapshot()
                self._on_state_change(old, new, snap)
            except Exception:
                pass  # fire-and-forget


# ─── Convenience factories ─────────────────────────────────────────────────────

def circuit_breaker(
    name: str,
    **kwargs,
) -> CircuitBreaker:
    """Create a CircuitBreaker with default config, passing any kwargs to CircuitBreakerConfig."""
    return CircuitBreaker(name=name, config=CircuitBreakerConfig(**kwargs))


# ─── Type variable for generics ───────────────────────────────────────────────

_T = ...  # type: ignore[assignment, misc]
