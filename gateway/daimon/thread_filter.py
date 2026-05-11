"""Thread ownership tracking — only creator + admins can trigger the agent."""
from __future__ import annotations

import logging
import threading
from typing import Optional

from gateway.daimon.config import DaimonConfig
from gateway.daimon.tier import resolve_tier

logger = logging.getLogger(__name__)


class ThreadOwnershipTracker:
    """Tracks which Discord user created which thread.

    Thread-safe. In-memory only (future: Discord API recovery on restart).
    Bounded to MAX_TRACKED threads to prevent unbounded memory growth.
    """

    MAX_TRACKED = 10_000  # Safety cap — well above 50 concurrent × 5/day/user

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owners: dict[str, str] = {}  # thread_id → creator_user_id

    def register(self, thread_id: str, creator_id: str) -> None:
        """Record that a user created a thread."""
        with self._lock:
            # Evict oldest entries if at capacity (simple FIFO via dict ordering)
            if len(self._owners) >= self.MAX_TRACKED and thread_id not in self._owners:
                # Remove oldest 10% to avoid evicting on every insert
                evict_count = self.MAX_TRACKED // 10
                for _ in range(evict_count):
                    try:
                        self._owners.pop(next(iter(self._owners)))
                    except (StopIteration, RuntimeError):
                        break
            self._owners[thread_id] = creator_id
        logger.debug("Registered thread %s owned by %s", thread_id, creator_id)

    def get_owner(self, thread_id: str) -> Optional[str]:
        """Get the creator of a thread, or None if unknown."""
        with self._lock:
            return self._owners.get(thread_id)

    def unregister(self, thread_id: str) -> None:
        """Remove tracking for a closed/archived thread."""
        with self._lock:
            self._owners.pop(thread_id, None)

    def should_process(self, author_id: str, thread_id: str, cfg: DaimonConfig, role_ids: Optional[list[str]] = None) -> bool:
        """Determine if a message from author_id in thread_id should be processed.

        Returns True if:
        - The author is an admin (always allowed)
        - The author is the thread creator
        - The thread is unknown (not tracked — e.g., pre-existing thread, allow through)
        """
        # Admins always get through
        tier = resolve_tier(author_id, cfg, role_ids=role_ids)
        if tier is not None and tier.is_admin:
            return True

        # If tier is None (user should be ignored), don't process
        if tier is None:
            return False

        # Check thread ownership
        owner = self.get_owner(thread_id)
        if owner is None:
            # Unknown thread — not daimon-managed, allow through
            # (regular Discord threads that existed before Daimon)
            return True

        return author_id == owner

    @property
    def tracked_count(self) -> int:
        """Number of threads currently tracked."""
        with self._lock:
            return len(self._owners)
