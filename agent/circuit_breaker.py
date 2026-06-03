"""
Per-feature circuit breakers (inspired by Claude Code architecture).

Each circuit has its own failure counter and max-failures threshold.
When tripped, the circuit blocks further attempts until explicitly reset
(typically at turn boundaries).

Unlike tool_guardrails (which monitors individual tool call patterns),
circuit breakers protect *features* — compression, delegation, provider
fallback, model retries — from cascading failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Mapping

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"       # normal operation
    OPEN = "open"           # tripped — blocks further attempts
    HALF_OPEN = "half_open" # testing if recovered


@dataclass
class CircuitBreaker:
    """Single circuit breaker instance."""

    name: str
    max_failures: int = 3
    reset_after_turns: int = 0  # 0 = never auto-reset (manual only)
    description: str = ""

    # Internal state
    _failures: int = field(default=0, repr=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, repr=False)
    _turn_count: int = field(default=0, repr=False)

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    @property
    def would_allow(self) -> bool:
        """Can this circuit accept new attempts?"""
        return self._state != CircuitState.OPEN

    def record_failure(self) -> bool:
        """Record a failure. Returns True if circuit just tripped."""
        self._failures += 1
        if self._failures >= self.max_failures and self._state == CircuitState.CLOSED:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker '%s' TRIPPED after %d failures (max=%d): %s",
                self.name, self._failures, self.max_failures, self.description,
            )
            return True
        return False

    def record_success(self):
        """Record a success — moves from HALF_OPEN to CLOSED."""
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failures = 0
            logger.info("Circuit breaker '%s' RECOVERED (half-open succeeded)", self.name)

    def reset(self):
        """Force-reset to closed state."""
        self._state = CircuitState.CLOSED
        self._failures = 0
        logger.debug("Circuit breaker '%s' manually reset", self.name)

    def try_reset(self) -> bool:
        """Attempt to move to half-open. Returns True if allowed to test."""
        if self._state == CircuitState.OPEN:
            if self.reset_after_turns > 0 and self._turn_count >= self.reset_after_turns:
                self._state = CircuitState.HALF_OPEN
                self._turn_count = 0
                logger.info("Circuit breaker '%s' → HALF_OPEN (auto-reset after %d turns)", self.name, self.reset_after_turns)
                return True
            return False
        return True

    def advance_turn(self):
        """Called at start of each turn."""
        self._turn_count += 1
        self.try_reset()


@dataclass
class CircuitBreakerPanel:
    """Collection of circuit breakers for different feature domains."""

    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)

    # Preset configurations matching Claude Code patterns
    PRESETS: ClassVar[Mapping[str, tuple[int, int, str]]] = {
        "compression":      (3, 0, "Auto-compact consecutive failures"),
        "delegation":       (3, 0, "delegate_task consecutive failures"),
        "provider_fallback":(3, 0, "Provider fallback chain exhausted"),
        "same_solution":    (2, 0, "Same solution retried"),
        "model_error":      (3, 0, "Model API errors (5xx/connection)"),
        "no_progress":      (5, 0, "No progress tool calls"),
        "api_retry":        (3, 0, "API retry exhaustion"),
    }

    def get(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        if name not in self.breakers:
            preset = self.PRESETS.get(name)
            if preset:
                max_f, reset_t, desc = preset
                self.breakers[name] = CircuitBreaker(
                    name=name, max_failures=max_f, reset_after_turns=reset_t, description=desc,
                )
            else:
                self.breakers[name] = CircuitBreaker(name=name)
        return self.breakers[name]

    def record_failure(self, name: str) -> bool:
        """Record a failure on a circuit. Returns True if circuit tripped."""
        return self.get(name).record_failure()

    def record_success(self, name: str):
        """Record a success on a circuit."""
        self.get(name).record_success()

    def would_allow(self, name: str) -> bool:
        """Check if a circuit allows new attempts."""
        return self.get(name).would_allow

    def is_open(self, name: str) -> bool:
        """Check if a circuit is tripped."""
        return self.get(name).is_open

    def reset(self, name: str):
        """Force-reset a circuit."""
        self.get(name).reset()

    def advance_turn(self):
        """Advance turn counters for all circuits."""
        for breaker in self.breakers.values():
            breaker.advance_turn()

    def reset_all(self):
        """Reset all circuits — called at start of each conversation turn."""
        for breaker in self.breakers.values():
            breaker.reset()

    def status(self) -> list[dict[str, Any]]:
        """Return status of all circuits."""
        return [
            {
                "name": b.name,
                "state": b.state.value,
                "failures": b.failures,
                "max_failures": b.max_failures,
                "description": b.description,
            }
            for b in self.breakers.values()
        ]

    def tripped_circuits(self) -> list[str]:
        """Return names of all open (tripped) circuits."""
        return [name for name, b in self.breakers.items() if b.is_open]
