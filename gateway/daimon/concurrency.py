"""Thread-safe session concurrency tracking for Daimon gateway."""

import threading
import time
from collections import deque
from typing import Optional


class ConcurrencyManager:
    """Thread-safe session concurrency tracking."""

    def __init__(self, max_active: int = 50, max_threads_per_day: int = 5):
        self._max_active = max_active
        self._max_threads_per_day = max_threads_per_day
        self._lock = threading.Lock()
        self._active: dict[str, str] = {}  # thread_id → user_id
        self._queue: deque[tuple[str, str]] = deque()  # FIFO of (thread_id, user_id)
        self._daily_usage: dict[str, list[float]] = {}  # user_id → list of timestamps

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def queue_length(self) -> int:
        with self._lock:
            return len(self._queue)

    def _prune_daily(self, user_id: str) -> None:
        """Remove timestamps older than 24h. Must be called with lock held."""
        if user_id not in self._daily_usage:
            return
        cutoff = time.time() - 86400
        self._daily_usage[user_id] = [
            ts for ts in self._daily_usage[user_id] if ts > cutoff
        ]

    def check_daily_limit(self, user_id: str) -> tuple[bool, str]:
        """Check if user has remaining daily allowance (rolling 24h window).

        Returns:
            (allowed, reason_if_denied) — reason is empty string if allowed.
        """
        with self._lock:
            self._prune_daily(user_id)
            usage = self._daily_usage.get(user_id, [])
            if len(usage) >= self._max_threads_per_day:
                return (
                    False,
                    f"Daily limit reached ({self._max_threads_per_day} threads per 24h)",
                )
            return (True, "")

    def try_acquire(self, thread_id: str, user_id: str) -> tuple[bool, int]:
        """Try to acquire an active slot.

        Records daily usage on successful acquisition.

        Returns:
            (acquired, queue_position) — queue_position is 0 if acquired.
        """
        with self._lock:
            # Idempotency: if thread already active, return success (no double-count)
            if thread_id in self._active:
                return (True, 0)

            # Check daily limit
            self._prune_daily(user_id)
            usage = self._daily_usage.get(user_id, [])
            if len(usage) >= self._max_threads_per_day:
                # Cannot even queue — daily limit hit
                return (False, 0)

            # Try to get an active slot
            if len(self._active) < self._max_active:
                self._active[thread_id] = user_id
                # Record daily usage
                if user_id not in self._daily_usage:
                    self._daily_usage[user_id] = []
                self._daily_usage[user_id].append(time.time())
                return (True, 0)

            # No active slot available — add to queue
            self._queue.append((thread_id, user_id))
            queue_position = len(self._queue)
            return (False, queue_position)

    def release(self, thread_id: str) -> Optional[str]:
        """Release an active slot and promote the next queued session.

        Also cleans the thread from the queue if it's there (early termination).

        Returns:
            The promoted thread_id, or None if nothing was promoted.
        """
        with self._lock:
            # Remove from active if present
            if thread_id in self._active:
                del self._active[thread_id]
            else:
                # Not in active — remove from queue (early termination)
                self._queue = deque(
                    (tid, uid) for tid, uid in self._queue if tid != thread_id
                )
                return None

            # Try to promote next from queue
            while self._queue:
                next_thread_id, next_user_id = self._queue.popleft()
                # Verify the promoted user still has daily allowance
                self._prune_daily(next_user_id)
                usage = self._daily_usage.get(next_user_id, [])
                if len(usage) < self._max_threads_per_day:
                    self._active[next_thread_id] = next_user_id
                    # Record daily usage for promoted session
                    if next_user_id not in self._daily_usage:
                        self._daily_usage[next_user_id] = []
                    self._daily_usage[next_user_id].append(time.time())
                    return next_thread_id

            return None
