"""
RecoveryEngine — Translates classification decisions into concrete actions.

The engine is a thin integration layer: it receives `ClassifiedError` objects
from the ErrorClassifier and dispatches the recovery action to the appropriate
gateway subsystem.  All heavy lifting (backoff, circuit breaker, persistence)
lives in the specialised resilience modules this engine calls.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, Optional, Set

from .error_classifier import (
    ClassifiedError,
    ErrorCategory,
    ErrorSeverity,
    RecoveryAction,
)

logger = logging.getLogger(__name__)


# ── In-memory deduplication ─────────────────────────────────────

@dataclass
class _DedupEntry:
    tb_hash: str
    first_seen: float
    count: int = 1


class _DedupRing:
    """Prevents retry storms by suppressing identical errors within a window."""

    def __init__(self, max_entries: int = 100, window_sec: float = 60.0):
        self._max = max_entries
        self._window = window_sec
        self._ring: Deque[_DedupEntry] = deque()

    def should_act(self, classified: ClassifiedError) -> bool:
        """Return True if we should act on this error (first occurrence in window)."""
        now = time.time()
        # Trim expired entries
        while self._ring and now - self._ring[0].first_seen > self._window:
            self._ring.popleft()

        # Find existing
        for entry in self._ring:
            if entry.tb_hash == classified.traceback_hash:
                entry.count += 1
                logger.debug(
                    "Dedup: suppressing %s (count=%d in %d s)",
                    classified.traceback_hash, entry.count, int(now - entry.first_seen),
                )
                return False

        self._ring.append(_DedupEntry(tb_hash=classified.traceback_hash, first_seen=now))
        if len(self._ring) > self._max:
            self._ring.popleft()
        return True


# ── Engine ──────────────────────────────────────────────────────

class RecoveryEngine:
    """Executes recovery actions determined by ErrorClassifier.classify().

    The engine delegates to gateway subsystems (circuit breaker, memory monitor,
    platform manager) and should be created once per GatewayRunner.
    """

    MAX_IMMEDIATE_RETRIES = 3

    def __init__(self):
        self._dedup = _DedupRing()
        self._circuit_breaker: Any = None     # set by GatewayRunner
        self._platform_manager: Any = None    # set by GatewayRunner
        self._checkpoint: Any = None          # set by GatewayRunner
        self._alert_hook: Any = None          # callable(ClassifiedError) -> None
        self._shutdown_callback: Optional[Any] = None  # set by GatewayRunner

        # Per-location retry counts (cleared on success)
        self._retry_counts: Dict[str, int] = {}

    # ── Public API ───────────────────────────────────────────────

    async def execute(self, classified: ClassifiedError) -> None:
        """Dispatch the classified error's recovery action."""

        if not self._dedup.should_act(classified):
            return  # duplicate within window — suppress

        action = classified.action
        logger.info(
            "RecoveryEngine: %s → %s (%s/%s)",
            classified.message[:120], action.name,
            classified.category.name, classified.severity.name,
        )

        try:
            if action == RecoveryAction.RETRY_IMMEDIATE:
                await self._retry_immediate(classified)
            elif action == RecoveryAction.RETRY_BACKOFF:
                await self._retry_backoff(classified)
            elif action == RecoveryAction.RETRY_NEVER:
                await self._alert_admin(classified)
            elif action == RecoveryAction.PAUSE_PLATFORM:
                await self._pause_platform(classified)
            elif action == RecoveryAction.CIRCUIT_BREAK:
                await self._circuit_break(classified)
            elif action == RecoveryAction.ALERT_ADMIN:
                await self._alert_admin(classified)
            elif action == RecoveryAction.SHUTDOWN_GRACEFUL:
                await self._graceful_shutdown(classified)
            elif action == RecoveryAction.IGNORE:
                pass  # explicitly allowed; logged at DEBUG
            else:
                logger.error("Unknown recovery action: %s", action)
        except Exception as exc:
            logger.critical(
                "Recovery engine itself failed executing %s: %s",
                action.name, exc, exc_info=True,
            )

    # ── Action implementations ───────────────────────────────────

    async def _retry_immediate(self, classified: ClassifiedError) -> None:
        """Signal that the caller should retry immediately (no sleep).

        This action is a *hint* — the actual retry must be implemented by the
        calling code that caught the exception.  If we've retried too many
        times for this location, escalate to backoff.
        """
        key = classified.context.location
        count = self._retry_counts.get(key, 0)
        if count >= self.MAX_IMMEDIATE_RETRIES:
            self._retry_counts.pop(key, None)
            # Escalate: re-classify as backoff rather than immediate
            await self._retry_backoff(classified)
            return
        self._retry_counts[key] = count + 1

    async def _retry_backoff(self, classified: ClassifiedError) -> None:
        """Escalate to the CircuitBreakerManager if available."""
        if self._circuit_breaker and classified.context.platform_id:
            await self._circuit_breaker.record_failure(
                classified.context.platform_id, classified.original_exception
            )

    async def _pause_platform(self, classified: ClassifiedError) -> None:
        if self._platform_manager and classified.context.platform_id:
            await self._platform_manager.pause_platform(
                classified.context.platform_id, auto=True
            )

    async def _circuit_break(self, classified: ClassifiedError) -> None:
        if self._circuit_breaker and classified.context.platform_id:
            await self._circuit_breaker.trip(classified.context.platform_id)

    async def _alert_admin(self, classified: ClassifiedError) -> None:
        if self._alert_hook:
            try:
                await self._alert_hook(classified)
            except Exception:
                logger.exception("Alert hook failed")

    async def _graceful_shutdown(self, classified: ClassifiedError) -> None:
        logger.critical(
            "SHUTDOWN triggered by %s: %s",
            classified.category.name, classified.message,
        )
        if self._checkpoint:
            await self._checkpoint.flush()
        # The gateway should have registered a shutdown_callback
        if hasattr(self, "_shutdown_callback"):
            await self._shutdown_callback(classified)

    def register_retry_success(self, location: str) -> None:
        """Clear retry count for a location when an operation succeeds."""
        self._retry_counts.pop(location, None)

    def set_circuit_breaker_manager(self, manager) -> None:
        """Wire the Phase 2 CircuitBreakerManager into the engine.

        The manager provides per-target circuit breakers via ``.get(target)``.
        When the engine receives a CIRCUIT_BREAK action it records a failure
        on the corresponding circuit breaker, which may transition it to OPEN.
        """
        self._circuit_breaker = manager
