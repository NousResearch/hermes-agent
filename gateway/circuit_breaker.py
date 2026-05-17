"""
CircuitBreaker — Stateful failure protection for gateway services.

Design
======
Three-state pattern (CLOSED → OPEN → HALF_OPEN → CLOSED) with
a sliding-window failure counter, configurable thresholds, and
thread-safe transitions.

Integration
===========
Wired into RecoveryEngine and platform reconnect logic. Circuit
breakers are keyed by target (e.g. "openrouter:claude-sonnet-4",
"telegram", "moonshot:kimi"). When a target trips OPEN, all calls
to that target are fast-failed with ``CircuitOpenError`` until
the cooldown expires and a trial call succeeds.

Config: ``resilience.circuit_breaker`` in ``config.yaml`` — see
the ``CircuitBreakerConfig`` dataclass below.

History
=======
  * Phase 2 addition — May 2026 — extracted from inline
    _pause_failed_platform / _resume_paused_platform logic in
    gateway/run.py into a standalone, reusable module.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ── Public exception ──────────────────────────────────────────────

class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted against an open (tripped) circuit breaker."""

    def __init__(self, target: str, tripped_at: float, cooldown_remaining: float):
        self.target = target
        self.tripped_at = tripped_at
        self.cooldown_remaining = cooldown_remaining
        super().__init__(
            f"Circuit for '{target}' is OPEN. "
            f"Tripped {tripped_at:.0f}s ago. "
            f"Cooldown remaining: {cooldown_remaining:.0f}s."
        )


# ── State Machine ─────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = auto()       # Normal — calls flow through, failures counted
    OPEN = auto()         # Tripped — all calls fast-fail
    HALF_OPEN = auto()    # Testing — limited trial calls allowed


@dataclass
class CircuitBreakerConfig:
    """Per-target configuration for a circuit breaker.

    All fields have sane defaults; override in config.yaml section
    ``resilience.circuit_breaker``.
    """

    failure_threshold: int = 5
    """Consecutive failures needed to trip from CLOSED → OPEN."""

    window_seconds: float = 60.0
    """Sliding window: failures older than this are aged out."""

    cooldown_seconds: float = 30.0
    """Time to stay OPEN before transitioning to HALF_OPEN."""

    half_open_max_calls: int = 1
    """Max trial calls allowed during HALF_OPEN before re-tripping."""

    half_open_window_seconds: float = 10.0
    """If a HALF_OPEN trial hasn't completed within this window, count it as failed."""

    enabled: bool = True
    """Per-target toggle; False = never trip, always CLOSED."""


# ── Failure Record ────────────────────────────────────────────────

@dataclass
class _FailureRecord:
    timestamp: float
    category: str = ""
    message: str = ""


# ── Circuit Breaker ───────────────────────────────────────────────

class CircuitBreaker:
    """Single-target circuit breaker with three-state machine.

    Thread-safe. Designed to be held inside CircuitBreakerManager
    and keyed by target name.

    Usage::

        breaker = CircuitBreaker("openrouter:claude-sonnet-4")
        with breaker:
            # ... make the call ...
            breaker.record_success()
        # on exception:
        breaker.record_failure(exc)
    """

    def __init__(
        self,
        target: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        self.target = target
        self.config = config or CircuitBreakerConfig()

        self._lock = threading.Lock()
        self._state: CircuitState = CircuitState.CLOSED
        self._failures: list[_FailureRecord] = []
        self._tripped_at: float = 0.0
        self._half_open_count: int = 0
        self._half_open_start: float = 0.0

        # Callbacks — set by CircuitBreakerManager
        self.on_state_change: Optional[Callable[[CircuitState, CircuitState, str], None]] = None

    # ── Properties ────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def is_open(self) -> bool:
        """True if the breaker is tripped (OPEN or HALF_OPEN with used-up trials)."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                return True
            if (self._state == CircuitState.HALF_OPEN
                    and self._half_open_count >= self.config.half_open_max_calls):
                return True
            return False

    @property
    def failure_count(self) -> int:
        with self._lock:
            self._prune_failures()
            return len(self._failures)

    @property
    def cooldown_remaining(self) -> float:
        """Seconds remaining until HALF_OPEN transition. 0 if not OPEN."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return 0.0
            elapsed = time.monotonic() - self._tripped_at
            return max(0.0, self.config.cooldown_seconds - elapsed)

    # ── Public API ────────────────────────────────────────────

    def record_success(self) -> None:
        """Called when a protected operation succeeds.

        Transitions HALF_OPEN → CLOSED on success (reset).
        No-op in CLOSED or OPEN states.
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                previous = self._state
                self._state = CircuitState.CLOSED
                self._failures.clear()
                self._half_open_count = 0
                self._tripped_at = 0.0
                self._half_open_start = 0.0
                logger.info(
                    "CircuitBreaker '%s' HALF_OPEN → CLOSED (trial succeeded)",
                    self.target,
                )
                if self.on_state_change:
                    self._safe_notify(previous, self._state, "trial_succeeded")

    def record_failure(self, exc: Optional[BaseException] = None) -> None:
        """Called when a protected operation fails.

        Increments the failure counter. If the threshold is reached
        in CLOSED state, transitions to OPEN.  If called in
        HALF_OPEN, transitions back to OPEN (re-tripped).
        """
        failure = _FailureRecord(
            timestamp=time.monotonic(),
            category=type(exc).__qualname__ if exc else "",
            message=str(exc)[:200] if exc else "",
        )
        with self._lock:
            self._prune_failures()
            self._failures.append(failure)

            if self._state == CircuitState.CLOSED:
                if len(self._failures) >= self.config.failure_threshold:
                    previous = self._state
                    self._state = CircuitState.OPEN
                    self._tripped_at = time.monotonic()
                    logger.warning(
                        "CircuitBreaker '%s' CLOSED → OPEN (%d failures in %.0fs)",
                        self.target,
                        len(self._failures),
                        self.config.window_seconds,
                    )
                    if self.on_state_change:
                        self._safe_notify(
                            previous, self._state,
                            "threshold_reached",
                        )
            elif self._state == CircuitState.HALF_OPEN:
                previous = self._state
                self._state = CircuitState.OPEN
                self._tripped_at = time.monotonic()
                self._half_open_count = 0
                self._half_open_start = 0.0
                logger.warning(
                    "CircuitBreaker '%s' HALF_OPEN → OPEN (trial failed)",
                    self.target,
                )
                if self.on_state_change:
                    self._safe_notify(previous, self._state, "trial_failed")

    def acquire(self) -> bool:
        """Request permission to make a call.

        Returns True if the call should proceed, False if it should be
        fast-failed (circuit is open).

        In HALF_OPEN, increments the trial counter.  If all trial
        slots are taken, returns False.

        *Callers must* call ``record_success()`` or
        ``record_failure()`` after a True return.
        """
        with self._lock:
            # Check and possibly transition states
            self._check_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                return False

            # HALF_OPEN
            if self._half_open_count >= self.config.half_open_max_calls:
                return False

            # Start trial window if first call
            if self._half_open_count == 0:
                self._half_open_start = time.monotonic()

            self._half_open_count += 1
            return True

    def __enter__(self) -> "CircuitBreaker":
        if not self.acquire():
            raise CircuitOpenError(
                self.target,
                self._tripped_at,
                self.cooldown_remaining,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None and issubclass(exc_type, CircuitOpenError):
            # CircuitOpenError is raised by acquire, not by the call itself.
            # Don't double-count — let the caller handle it.
            return False
        if exc_type is not None and exc_val is not None:
            self.record_failure(exc_val)
        else:
            self.record_success()
        return False  # don't suppress

    def reset(self) -> None:
        """Force the breaker back to CLOSED — manual intervention escape hatch."""
        with self._lock:
            previous = self._state
            self._state = CircuitState.CLOSED
            self._failures.clear()
            self._tripped_at = 0.0
            self._half_open_count = 0
            self._half_open_start = 0.0
            logger.info(
                "CircuitBreaker '%s' manually reset → CLOSED (was %s)",
                self.target,
                previous.name,
            )
            if self.on_state_change:
                self._safe_notify(previous, self._state, "manual_reset")

    # ── Internal ──────────────────────────────────────────────

    def _check_transition(self) -> None:
        """Evaluate state transitions that don't require a call event."""
        if self._state != CircuitState.OPEN:
            return

        # Check if cooldown has expired
        elapsed = time.monotonic() - self._tripped_at
        if elapsed >= self.config.cooldown_seconds:
            previous = self._state
            self._state = CircuitState.HALF_OPEN
            self._half_open_count = 0
            self._half_open_start = 0.0
            logger.info(
                "CircuitBreaker '%s' OPEN → HALF_OPEN (cooldown: %.0fs)",
                self.target,
                elapsed,
            )
            if self.on_state_change:
                self._safe_notify(previous, self._state, "cooldown_expired")

        # Expire stale HALF_OPEN trials
        if (self._state == CircuitState.HALF_OPEN
                and self._half_open_count > 0
                and self._half_open_start > 0):
            elapsed = time.monotonic() - self._half_open_start
            if elapsed >= self.config.half_open_window_seconds:
                previous = self._state
                self._state = CircuitState.OPEN
                self._tripped_at = time.monotonic()
                self._half_open_count = 0
                self._half_open_start = 0.0
                logger.warning(
                    "CircuitBreaker '%s' HALF_OPEN → OPEN (trial timed out after %.0fs)",
                    self.target,
                    elapsed,
                )
                if self.on_state_change:
                    self._safe_notify(previous, self._state, "trial_timeout")

    def _prune_failures(self) -> None:
        """Remove failure records outside the sliding window."""
        now = time.monotonic()
        cutoff = now - self.config.window_seconds
        self._failures = [f for f in self._failures if f.timestamp >= cutoff]

    def _safe_notify(self, old_state, new_state, reason):
        """Call on_state_change without leaking exceptions into the caller."""
        try:
            self.on_state_change(old_state, new_state, reason)  # type: ignore[misc]
        except Exception as _e:
            logger.debug(
                "on_state_change callback for '%s' failed: %s",
                self.target, _e,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serializable snapshot for status endpoints or /platform list."""
        with self._lock:
            self._prune_failures()
            return {
                "target": self.target,
                "state": self._state.name,
                "failure_count": len(self._failures),
                "threshold": self.config.failure_threshold,
                "cooldown_remaining": self.cooldown_remaining,
                "cooldown_seconds": self.config.cooldown_seconds,
                "enabled": self.config.enabled,
            }


# ── Circuit Breaker Manager ───────────────────────────────────────

class CircuitBreakerManager:
    """Collection of named circuit breakers with shared configuration.

    The manager is the single entry point for the gateway.  Create it
    once in ``start_gateway``, wire it into ``RecoveryEngine``, and
    pass it to platform adapters that need call-by-call protection.

    Usage::

        cbm = CircuitBreakerManager()
        breaker = cbm.get("openrouter:claude-sonnet-4")
        try:
            with breaker:
                response = await provider.send(...)
        except CircuitOpenError:
            # fast-fail — circuit is open
            ...
    """

    def __init__(
        self,
        default_config: Optional[CircuitBreakerConfig] = None,
        per_target_overrides: Optional[Dict[str, CircuitBreakerConfig]] = None,
    ) -> None:
        self.default_config = default_config or CircuitBreakerConfig()
        self._overrides = per_target_overrides or {}
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

        # Callback fired when any breaker changes state
        self.on_any_state_change: Optional[Callable[[str, CircuitState, CircuitState, str], None]] = None

    def get(self, target: str) -> CircuitBreaker:
        """Get or create a circuit breaker for *target*.

        Target naming convention: ``"provider:model"`` for LLM
        providers, ``"platform:name"`` for messaging platforms,
        or any unique string identifying the call target.
        """
        with self._lock:
            if target in self._breakers:
                return self._breakers[target]

            config = self._overrides.get(
                target,
                self.default_config,
            )
            breaker = CircuitBreaker(target, config=config)

            # Latch our state-change aggregator
            def _on_change(old, new, reason, t=target):  # type: ignore
                if self.on_any_state_change:
                    try:
                        self.on_any_state_change(t, old, new, reason)
                    except Exception as _e:
                        logger.debug(
                            "CircuitBreakerManager.on_any_state_change failed: %s", _e,
                        )

            breaker.on_state_change = _on_change
            self._breakers[target] = breaker
            return breaker

    def get_or_none(self, target: str) -> Optional[CircuitBreaker]:
        """Return the breaker for *target* without creating one."""
        with self._lock:
            return self._breakers.get(target)

    def reset_all(self) -> None:
        """Force all breakers back to CLOSED — emergency escape hatch."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of all circuit breaker states."""
        with self._lock:
            return {
                target: breaker.to_dict()
                for target, breaker in self._breakers.items()
            }

    def prune_stale(self, max_age_seconds: float = 3600.0) -> int:
        """Remove breakers that have been idle (CLOSED, no failures)
        for *max_age_seconds*. Returns count removed."""
        # Implementation note: we don't currently track idle time
        # per-breaker, so for now this is a no-op placeholder that
        # Phase 3 (demerging run.py) will flesh out.
        return 0
