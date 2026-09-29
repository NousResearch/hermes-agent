"""
LLM Circuit Breaker Manager — Cascading Failure Prevention for Hermes Agent.

Provides per-provider circuit breakers that automatically detect when a provider
is failing and circuit-trip to prevent wasted retries.  Integrates with the
fallback chain so that when a circuit opens, the agent immediately tries the
next provider without burning retries on a known-bad endpoint.

Design principles:
  GL (Generative Loop)  — self-healing: OPEN→HALF_OPEN→CLOSED auto-recovers
  OS (Observable State)  — all state via snapshot(), emitted as EventBus events
  MI (Module Independence) — pure-Python, no external deps
  LP (Least Privilege)   — per-provider circuits; one bad actor doesn't affect others

Usage::

    from agent.llm_circuit_breakers import LLMCircuitBreakerManager

    manager = LLMCircuitBreakerManager()

    # Before making an LLM call:
    breaker = manager.get_breaker(provider="openrouter", model="claude-3.5-sonnet")
    result = breaker.call(
        my_llm_call_function, *args,
        fallback_factory=lambda exc: None,  # return None to trigger fallback
        record_failure=True,
    )
    if result is None:
        # Circuit is OPEN or call failed — activate fallback
        activate_next_fallback_provider()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional, Tuple

from agent.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerSnapshot,
    CircuitState,
)

logger = logging.getLogger(__name__)

# ─── Default configuration ─────────────────────────────────────────────────────

# Conservative defaults: trip after 3 consecutive failures, recover after 30s.
# This balances protecting against cascading failures vs. not over-reacting.
DEFAULT_LLM_CB_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
    half_open_max_calls=1,
    success_threshold=1,
    excluded_exceptions=(ValueError, TypeError),  # Client errors don't trip circuit
)


def _breaker_name(provider: str, model: str) -> str:
    """Build a canonical circuit breaker name from provider + model."""
    return f"llm:{provider}:{model}"


# ─── Manager ─────────────────────────────────────────────────────────────────

class LLMCircuitBreakerManager:
    """
    Per-provider circuit breaker registry for LLM calls.

    Manages a thread-safe dict of CircuitBreaker instances, one per
    (provider, model) pair.  Provides a clean interface for the agent
    to check circuit state before making calls and record outcomes after.

    Events emitted (via optional EventBus):
      - circuit.open   — circuit tripped to OPEN state
      - circuit.half_open — circuit transitioned to HALF_OPEN (recovery probe)
      - circuit.closed — circuit recovered to CLOSED state
      - circuit.rejected — call rejected because circuit was OPEN

    All events carry a snapshot payload with current state.
    """

    def __init__(
        self,
        default_config: CircuitBreakerConfig = None,
        event_bus=None,
    ):
        """
        Args:
            default_config: Default thresholds for new breakers.
            event_bus: Optional EventBus for emitting state-change events.
        """
        self._default_config = default_config or DEFAULT_LLM_CB_CONFIG
        self._breakers: Dict[Tuple[str, str], CircuitBreaker] = {}
        self._lock = threading.RLock()
        self._event_bus = event_bus

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_breaker(
        self,
        provider: str,
        model: str,
        config: CircuitBreakerConfig = None,
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker for the given provider/model.

        Thread-safe.  Returns the existing breaker if one is already
        registered for this (provider, model) pair.
        """
        key = (provider, model)
        with self._lock:
            if key not in self._breakers:
                cfg = config or self._default_config
                cb = CircuitBreaker(
                    name=_breaker_name(provider, model),
                    config=cfg,
                    on_state_change=self._on_state_change,
                    on_rejected=self._on_rejected,
                )
                self._breakers[key] = cb
            return self._breakers[key]

    def get_or_wrap(
        self,
        provider: str,
        model: str,
        func: Callable,
        *args,
        record_failure: bool = True,
        fallback_factory: Callable[[Exception], object] = None,
        **kwargs,
    ):
        """
        Convenience: get the breaker and call through it in one step.

        Returns the function's result, or the fallback_factory output if
        the circuit is OPEN or the call raises a non-excluded exception
        (when record_failure=True).
        """
        breaker = self.get_breaker(provider, model)
        return breaker.call(
            func, *args,
            record_failure=record_failure,
            fallback_factory=fallback_factory,
            **kwargs,
        )

    def snapshot_all(self) -> Dict[Tuple[str, str], CircuitBreakerSnapshot]:
        """
        Return snapshots for all registered breakers.

        Thread-safe snapshot of the entire circuit breaker registry.
        """
        with self._lock:
            return {key: cb.snapshot() for key, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all breakers to CLOSED.  For testing or admin recovery."""
        with self._lock:
            for cb in self._breakers.values():
                cb.reset()
        logger.info("LLMCircuitBreakerManager: all circuits reset")

    def is_provider_available(self, provider: str) -> bool:
        """
        Return True if at least one breaker for this provider is not OPEN.

        Checks all registered breakers for the given provider.  Returns
        False only if ALL breakers for this provider are in OPEN state.
        """
        with self._lock:
            for (p, _), cb in self._breakers.items():
                if p == provider and cb.state != CircuitState.OPEN:
                    return True
        return False

    def get_open_breakers(self) -> list[CircuitBreakerSnapshot]:
        """Return snapshots of all breakers currently in OPEN state."""
        snapshots = self.snapshot_all()
        return [s for s in snapshots.values() if s.is_open]

    # ── Internal event handlers ───────────────────────────────────────────────

    def _on_state_change(
        self,
        old: CircuitState,
        new: CircuitState,
        snap: CircuitBreakerSnapshot,
    ) -> None:
        """Handle circuit state transitions — emit EventBus events."""
        # Parse provider/model from breaker name (format: "llm:provider:model")
        name_parts = snap.name.split(":", 2)
        provider = name_parts[1] if len(name_parts) >= 2 else "unknown"
        model = name_parts[2] if len(name_parts) >= 3 else "unknown"

        payload = {
            "provider": provider,
            "model": model,
            "old_state": old.name,
            "new_state": new.name,
            "failure_count": snap.failure_count,
            "last_failure_reason": snap.last_failure_reason,
            "open_until": snap.open_until,
            "total_calls": snap.total_calls,
            "total_failures": snap.total_failures,
            "total_rejected": snap.total_rejected,
        }

        if new == CircuitState.OPEN:
            self._emit("circuit.open", payload)
            wait = max(0.0, (snap.open_until or 0) - time.monotonic())
            logger.warning(
                "LLM circuit OPEN: provider=%s model=%s (failures=%d, retry in %.1fs)",
                provider, model, snap.failure_count, wait,
            )
        elif new == CircuitState.HALF_OPEN:
            self._emit("circuit.half_open", payload)
            logger.info(
                "LLM circuit HALF_OPEN: provider=%s model=%s",
                provider, model,
            )
        elif new == CircuitState.CLOSED:
            self._emit("circuit.closed", payload)
            logger.info(
                "LLM circuit CLOSED: provider=%s model=%s (recovered after %.1fs)",
                provider, model,
                time.monotonic() - ((snap.last_failure_time or time.monotonic()) - (snap.open_until or 0)),
            )

    def _on_rejected(self, exc: CircuitBreakerOpen) -> None:
        """Handle rejected calls — emit EventBus event."""
        name_parts = exc.resource_name.split(":", 2)
        provider = name_parts[1] if len(name_parts) >= 2 else "unknown"
        model = name_parts[2] if len(name_parts) >= 3 else "unknown"

        self._emit("circuit.rejected", {
            "provider": provider,
            "model": model,
            "open_until": exc.open_until,
            "wait_remaining": max(0.0, exc.open_until - time.monotonic()),
        })
        logger.debug(
            "LLM call rejected by OPEN circuit: provider=%s model=%s",
            provider, model,
        )

    def _emit(self, event_type: str, payload: dict) -> None:
        """Emit an EventBus event (fire-and-forget)."""
        if self._event_bus is None:
            return
        try:
            from agent.hermes.analytics import Event
            event = Event(type=event_type, payload=payload)
            self._event_bus.emit(event)
        except Exception as exc:
            logger.debug("LLMCircuitBreakerManager: EventBus emit failed: %s", exc)


# ─── Singleton ────────────────────────────────────────────────────────────────

_manager: Optional[LLMCircuitBreakerManager] = None
_manager_lock = threading.Lock()


def get_circuit_breaker_manager(event_bus=None) -> LLMCircuitBreakerManager:
    """
    Get or create the global LLMCircuitBreakerManager singleton.
    """
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = LLMCircuitBreakerManager(event_bus=event_bus)
        elif event_bus is not None:
            # Only update _event_bus if still None (double-checked locking)
            if _manager._event_bus is None:
                _manager._event_bus = event_bus
    return _manager


def reset_circuit_breaker_manager() -> None:
    """Reset the global singleton (for testing)."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.reset_all()
        _manager = None
