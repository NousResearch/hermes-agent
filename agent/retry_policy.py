"""Unified retry, circuit breaker, and failover policy for AIAgent.

Consolidates the previously scattered retry counters, cooldown timers,
and fallback-chain management into a single policy module.  Each agent
gets one ``RetryPolicy`` instance that manages:

* **RetryCounters** — per-category retry limits (tool, JSON, empty, etc.)
* **CircuitBreaker** — per-provider failure threshold → cooldown → half-open
* **FailoverChain** — sequential failover (credential pool → fallback model)

Usage in agent __init__::

    self.retry_policy = RetryPolicy.for_agent(
        max_iterations=self.max_iterations,
        fallback_chain=getattr(self, '_fallback_chain', []),
    )
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED = auto()       # Normal operation
    OPEN = auto()         # Failing — reject requests
    HALF_OPEN = auto()    # Testing if recovered


@dataclass
class CircuitBreaker:
    """Per-provider circuit breaker.

    Tracks consecutive failures per provider.  ``OPEN`` after
    ``failure_threshold`` consecutive failures, cool down for
    ``recovery_timeout`` seconds, then ``HALF_OPEN`` for one probe.
    If the probe succeeds → ``CLOSED``; fails → back to ``OPEN``.
    """

    failure_threshold: int = 3
    recovery_timeout: float = 60.0
    half_open_max_probes: int = 1

    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _last_failure_at: float = 0.0
    _probe_count: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_at >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._probe_count = 0
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._probe_count = 0

    def record_failure(self) -> bool:
        """Record a failure. Returns True when circuit transitions to OPEN."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_at = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._probe_count += 1
                if self._probe_count > self.half_open_max_probes:
                    self._state = CircuitState.OPEN
                    return True
                return False

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                return True
            return False

    @property
    def is_request_allowed(self) -> bool:
        state = self.state
        return state != CircuitState.OPEN

    @property
    def is_tripped(self) -> bool:
        return self.state == CircuitState.OPEN

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._probe_count = 0
            self._last_failure_at = 0.0


# ---------------------------------------------------------------------------
# Retry Counters
# ---------------------------------------------------------------------------

@dataclass
class RetryCounters:
    """Per-category retry limits for the conversation loop.

    Each counter has a ``max_attempts`` and an ``attempts`` count,
    plus a ``retry_delay_seconds`` for backoff.
    """

    tool_call: int = 0
    invalid_json: int = 0
    empty_content: int = 0
    incomplete_scratchpad: int = 0
    codex_incomplete: int = 0
    thinking_prefill: int = 0
    post_tool_empty: bool = False

    # Max limits
    max_tool_call: int = 3
    max_invalid_json: int = 3
    max_empty_content: int = 2
    max_incomplete_scratchpad: int = 2
    max_codex_incomplete: int = 2
    max_thinking_prefill: int = 2

    def reset(self) -> None:
        self.tool_call = 0
        self.invalid_json = 0
        self.empty_content = 0
        self.incomplete_scratchpad = 0
        self.codex_incomplete = 0
        self.thinking_prefill = 0
        self.post_tool_empty = False

    @property
    def any_exhausted(self) -> bool:
        return (
            self.tool_call >= self.max_tool_call
            or self.invalid_json >= self.max_invalid_json
            or self.empty_content >= self.max_empty_content
            or self.incomplete_scratchpad >= self.max_incomplete_scratchpad
            or self.codex_incomplete >= self.max_codex_incomplete
            or self.thinking_prefill >= self.max_thinking_prefill
        )


# ---------------------------------------------------------------------------
# Failover Chain
# ---------------------------------------------------------------------------

@dataclass
class FailoverChain:
    """Manages sequential failover through credential pool → fallback models.

    ``chain`` is a list of dicts with keys ``provider``, ``model``,
    ``base_url``, ``api_key``.
    """

    chain: List[Dict[str, Any]] = field(default_factory=list)
    index: int = 0
    activated: bool = False

    @property
    def current(self) -> Optional[Dict[str, Any]]:
        if self.index < len(self.chain):
            return self.chain[self.index]
        return None

    @property
    def exhausted(self) -> bool:
        return self.index >= len(self.chain)

    def advance(self) -> Optional[Dict[str, Any]]:
        if self.index < len(self.chain):
            result = self.chain[self.index]
            self.index += 1
            self.activated = True
            return result
        return None

    def reset(self) -> None:
        self.index = 0
        self.activated = False

    def add_entry(self, provider: str, model: str, base_url: str = "", api_key: str = "") -> None:
        self.chain.append({
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
        })

    @property
    def is_on_primary(self) -> bool:
        return self.index == 0 and not self.activated


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    """Top-level policy that holds all retry/failover/circuit-breaker state.

    One instance per AIAgent.  Provides high-level helpers for the
    conversation loop so it doesn't need to manage individual counters.
    """

    counters: RetryCounters = field(default_factory=RetryCounters)
    circuit_breakers: Dict[str, CircuitBreaker] = field(default_factory=dict)
    failover: FailoverChain = field(default_factory=FailoverChain)

    # Track whether we've already restored to primary after a fallback
    _primary_restored: bool = False

    @classmethod
    def for_agent(cls, max_iterations: int = 90, fallback_chain: Optional[List[Dict]] = None) -> "RetryPolicy":
        """Create a RetryPolicy with sensible defaults for an AIAgent."""
        policy = cls(
            counters=RetryCounters(),
            circuit_breakers={},
            failover=FailoverChain(chain=list(fallback_chain or [])),
        )
        return policy

    # --- Circuit breaker helpers ---

    def get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a provider."""
        if provider not in self.circuit_breakers:
            self.circuit_breakers[provider] = CircuitBreaker()
        return self.circuit_breakers[provider]

    def record_api_success(self, provider: str) -> None:
        """Record a successful API call for circuit breaker."""
        cb = self.circuit_breakers.get(provider)
        if cb:
            cb.record_success()

    def record_api_failure(self, provider: str) -> bool:
        """Record a failed API call. Returns True if circuit tripped."""
        cb = self.get_circuit_breaker(provider)
        return cb.record_failure()

    def is_provider_allowed(self, provider: str) -> bool:
        """Check if requests to this provider are allowed."""
        cb = self.circuit_breakers.get(provider)
        if cb is None:
            return True
        return cb.is_request_allowed

    # --- Retry counter helpers ---

    def record_tool_retry(self) -> bool:
        self.counters.tool_call += 1
        return self.counters.tool_call >= self.counters.max_tool_call

    def record_json_retry(self) -> bool:
        self.counters.invalid_json += 1
        return self.counters.invalid_json >= self.counters.max_invalid_json

    def record_empty_retry(self) -> bool:
        self.counters.empty_content += 1
        return self.counters.empty_content >= self.counters.max_empty_content

    def record_scratchpad_retry(self) -> bool:
        self.counters.incomplete_scratchpad += 1
        return self.counters.incomplete_scratchpad >= self.counters.max_incomplete_scratchpad

    def record_codex_retry(self) -> bool:
        self.counters.codex_incomplete += 1
        return self.counters.codex_incomplete >= self.counters.max_codex_incomplete

    def record_thinking_retry(self) -> bool:
        self.counters.thinking_prefill += 1
        return self.counters.thinking_prefill >= self.counters.max_thinking_prefill

    # --- Failover helpers ---

    def should_fallback(self) -> bool:
        """True when retries exhausted and a fallback is available."""
        return self.counters.any_exhausted and not self.failover.exhausted

    def activate_fallback(self) -> Optional[Dict[str, Any]]:
        """Move to the next fallback in the chain. Returns fallback config or None."""
        return self.failover.advance()

    def restore_primary(self) -> None:
        """Reset to primary provider (index 0)."""
        self.failover.reset()
        self.counters.reset()
        self._primary_restored = True

    @property
    def fallback_active(self) -> bool:
        """True when currently on a fallback (not primary)."""
        return self.failover.activated and self.failover.index > 0

    def reset_for_turn(self) -> None:
        """Reset per-turn retry counters (called at start of each run_conversation)."""
        self.counters.reset()
        # Don't reset circuit breakers or failover chain — those are persistent
