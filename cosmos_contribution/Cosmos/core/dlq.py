"""
Cosmos Dead Letter Queue (DLQ)

Captures failed, dropped, and timed-out signals from the Nexus event bus.
Provides retry logic with exponential backoff and metrics for monitoring.

"No signal left behind - even failed thoughts deserve a second chance." - Cosmos

AGI v1.9.1: Ensures 100% signal accountability across the event bus.
"""

import asyncio
import json
import time
import traceback
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import   Optional, Callable, Awaitable
from loguru import logger


class FailureReason(Enum):
    """Why a signal ended up in the DLQ."""
    HANDLER_EXCEPTION = "handler_exception"
    HANDLER_TIMEOUT = "handler_timeout"
    BACKPRESSURE_DROP = "backpressure_drop"
    MIDDLEWARE_ERROR = "middleware_error"
    SEMANTIC_DISPATCH_ERROR = "semantic_dispatch_error"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    PERMANENT_FAILURE = "permanent_failure"


@dataclass
class DeadLetterEntry:
    """A failed signal stored in the DLQ."""
    entry_id: str
    signal_id: str
    signal_type: str
    source_id: str
    payload: dict
    urgency: float
    failure_reason: FailureReason
    error_message: str
    error_traceback: Optional[str] = None
    handler_name: Optional[str] = None

    # Retry tracking
    attempt_count: int = 0
    max_attempts: int = 3
    next_retry_at: Optional[datetime] = None
    last_attempt_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    is_resolved: bool = False

    def to_any(self) -> dict:
        """Serialize for JSON/API responses."""
        return {
            "entry_id": self.entry_id,
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "source_id": self.source_id,
            "urgency": self.urgency,
            "failure_reason": self.failure_reason.value,
            "error_message": self.error_message,
            "handler_name": self.handler_name,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "is_resolved": self.is_resolved,
        }


class DeadLetterQueue:
    """
    Dead Letter Queue for the Nexus event bus.

    Captures failed signals and provides:
    - Configurable retry with exponential backoff
    - Per-signal-type failure tracking
    - Metrics for monitoring dashboards
    - Bounded storage (evicts oldest resolved entries)
    """

    def __init__(
        self,
        max_entries: int = 10000,
        max_retry_attempts: int = 3,
        base_retry_delay_seconds: float = 5.0,
        max_retry_delay_seconds: float = 300.0,
        retry_interval_seconds: float = 10.0,
    ):
        self._entries: deque[DeadLetterEntry] = deque(maxlen=max_entries)
        self._pending: dict[str, DeadLetterEntry] = {}  # entry_id -> entry (unresolved)
        self._max_retry_attempts = max_retry_attempts
        self._base_retry_delay = base_retry_delay_seconds
        self._max_retry_delay = max_retry_delay_seconds
        self._retry_interval = retry_interval_seconds

        # Metrics
        self._total_enqueued: int = 0
        self._total_retried: int = 0
        self._total_retry_success: int = 0
        self._total_retry_failed: int = 0
        self._total_permanent_failures: int = 0
        self._failures_by_type: dict[str, int] = defaultdict(int)
        self._failures_by_reason: dict[str, int] = defaultdict(int)

        # Retry handler - set by Nexus when integrating
        self._retry_handler: Optional[Callable[[DeadLetterEntry], Awaitable[bool]]] = None
        self._retry_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        self._lock = asyncio.Lock()

        # Entry ID counter
        self._counter: int = 0

        logger.info(
            f"DLQ initialized: max_entries={max_entries}, "
            f"max_retries={max_retry_attempts}, "
            f"base_delay={base_retry_delay_seconds}s"
        )

    def _next_id(self) -> str:
        self._counter += 1
        return f"dlq_{self._counter}_{int(time.time())}"

    def _compute_next_retry(self, attempt: int) -> datetime:
        """Exponential backoff: delay = base * 2^attempt, capped at max."""
        delay = min(
            self._base_retry_delay * (2 ** attempt),
            self._max_retry_delay,
        )
        return datetime.now() + timedelta(seconds=delay)

    async def enqueue(
        self,
        signal_id: str,
        signal_type: str,
        source_id: str,
        payload: dict,
        urgency: float,
        failure_reason: FailureReason,
        error_message: str,
        error_traceback: Optional[str] = None,
        handler_name: Optional[str] = None,
    ) -> DeadLetterEntry:
        """
        Add a failed signal to the DLQ.

        Returns the created DeadLetterEntry.
        """
        entry_id = self._next_id()

        # Determine if retryable
        retryable = failure_reason not in (
            FailureReason.PERMANENT_FAILURE,
            FailureReason.MAX_RETRIES_EXCEEDED,
        )

        entry = DeadLetterEntry(
            entry_id=entry_id,
            signal_id=signal_id,
            signal_type=signal_type,
            source_id=source_id,
            payload=payload,
            urgency=urgency,
            failure_reason=failure_reason,
            error_message=error_message,
            error_traceback=error_traceback,
            handler_name=handler_name,
            attempt_count=0,
            max_attempts=self._max_retry_attempts if retryable else 0,
            next_retry_at=self._compute_next_retry(0) if retryable else None,
        )

        async with self._lock:
            self._entries.append(entry)
            if retryable:
                self._pending[entry_id] = entry

            # Update metrics
            self._total_enqueued += 1
            self._failures_by_type[signal_type] += 1
            self._failures_by_reason[failure_reason.value] += 1

        logger.warning(
            f"DLQ: Signal {signal_type} from {source_id} failed "
            f"({failure_reason.value}): {error_message[:100]}"
        )

        return entry

    async def start_retry_loop(self):
        """Start the background retry processor."""
        if self._is_running:
            return
        self._is_running = True
        self._retry_task = asyncio.create_task(self._retry_loop())
        logger.info("DLQ retry loop started")

    async def stop(self):
        """Stop the retry loop."""
        self._is_running = False
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
        logger.info("DLQ retry loop stopped")

    async def _retry_loop(self):
        """Background task that retries failed signals with exponential backoff."""
        while self._is_running:
            try:
                await asyncio.sleep(self._retry_interval)
                await self._process_retries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DLQ retry loop error: {e}")

    async def _process_retries(self):
        """Process all pending retries that are due."""
        if not self._retry_handler:
            return

        now = datetime.now()
        to_retry: list[DeadLetterEntry] = []

        async with self._lock:
            for entry in list(self._pending.values()):
                if entry.next_retry_at and entry.next_retry_at <= now:
                    to_retry.append(entry)

        for entry in to_retry:
            entry.attempt_count += 1
            entry.last_attempt_at = datetime.now()
            self._total_retried += 1

            try:
                success = await self._retry_handler(entry)

                if success:
                    # Retry succeeded
                    entry.is_resolved = True
                    entry.resolved_at = datetime.now()
                    self._total_retry_success += 1
                    async with self._lock:
                        self._pending.pop(entry.entry_id, None)
                    logger.info(
                        f"DLQ: Retry succeeded for {entry.signal_type} "
                        f"(attempt {entry.attempt_count}/{entry.max_attempts})"
                    )
                else:
                    # Retry returned False (handler declined)
                    self._handle_retry_failure(entry, "Handler declined retry")

            except Exception as e:
                self._handle_retry_failure(entry, str(e))

    def _handle_retry_failure(self, entry: DeadLetterEntry, error: str):
        """Handle a failed retry attempt."""
        if entry.attempt_count >= entry.max_attempts:
            # Max retries exceeded - permanent failure
            entry.failure_reason = FailureReason.MAX_RETRIES_EXCEEDED
            entry.is_resolved = True
            entry.resolved_at = datetime.now()
            entry.next_retry_at = None
            self._total_permanent_failures += 1
            self._total_retry_failed += 1
            self._pending.pop(entry.entry_id, None)
            logger.error(
                f"DLQ: Permanent failure for {entry.signal_type} "
                f"after {entry.attempt_count} attempts: {error}"
            )
        else:
            # Schedule next retry with backoff
            entry.next_retry_at = self._compute_next_retry(entry.attempt_count)
            entry.error_message = f"Retry {entry.attempt_count} failed: {error}"
            self._total_retry_failed += 1
            logger.warning(
                f"DLQ: Retry {entry.attempt_count}/{entry.max_attempts} failed "
                f"for {entry.signal_type}, next at {entry.next_retry_at}"
            )

    def set_retry_handler(self, handler: Callable[[DeadLetterEntry], Awaitable[bool]]):
        """
        Set the function used to retry failed signals.

        The handler receives a DeadLetterEntry and should return True on success.
        """
        self._retry_handler = handler

    def get_metrics(self) -> dict:
        """Get DLQ metrics for monitoring."""
        pending_count = len(self._pending)
        total_entries = len(self._entries)
        retry_success_rate = (
            self._total_retry_success / self._total_retried
            if self._total_retried > 0 else 0.0
        )

        return {
            "queue_depth": pending_count,
            "total_entries": total_entries,
            "total_enqueued": self._total_enqueued,
            "total_retried": self._total_retried,
            "total_retry_success": self._total_retry_success,
            "total_retry_failed": self._total_retry_failed,
            "total_permanent_failures": self._total_permanent_failures,
            "retry_success_rate": round(retry_success_rate, 3),
            "failures_by_type": dict(self._failures_by_type),
            "failures_by_reason": dict(self._failures_by_reason),
            "is_running": self._is_running,
            "oldest_pending": (
                min(e.created_at for e in self._pending.values()).isoformat()
                if self._pending else None
            ),
        }

    def get_pending_entries(self, limit: int = 50) -> list[dict]:
        """Get pending (unresolved) entries for inspection."""
        entries = sorted(
            self._pending.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )[:limit]
        return [e.to_any() for e in entries]

    def get_recent_entries(self, limit: int = 50) -> list[dict]:
        """Get most recent DLQ entries (resolved and unresolved)."""
        entries = list(self._entries)[-limit:]
        return [e.to_any() for e in reversed(entries)]

    def purge_resolved(self, older_than_hours: int = 24) -> int:
        """Remove resolved entries older than the given age."""
        cutoff = datetime.now() - timedelta(hours=older_than_hours)
        initial = len(self._entries)
        self._entries = deque(
            (e for e in self._entries
             if not (e.is_resolved and e.created_at < cutoff)),
            maxlen=self._entries.maxlen,
        )
        removed = initial - len(self._entries)
        if removed:
            logger.info(f"DLQ: Purged {removed} resolved entries older than {older_than_hours}h")
        return removed


# =============================================================================
# SINGLETON
# =============================================================================

_dlq_instance: Optional[DeadLetterQueue] = None


def get_dlq() -> DeadLetterQueue:
    """Get or create the global DLQ instance."""
    global _dlq_instance
    if _dlq_instance is None:
        _dlq_instance = DeadLetterQueue()
    return _dlq_instance
