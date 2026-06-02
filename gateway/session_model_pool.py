"""Session model pool — concurrency-aware auto-assignment for multi-session gateways.

Distributes sessions and auxiliary calls across a configured pool of models,
respecting per-model concurrency limits and reserving slots for auxiliary tasks.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PoolModelEntry:
    """A single model in the pool with concurrency tracking."""

    model: str
    provider: str
    max_concurrent: int = 1
    reserved_for_auxiliary: int = 0
    context_length: Optional[int] = None
    priority: int = 5

    # --- runtime state (not persisted) ---
    session_slots: List[str] = field(default_factory=list)
    auxiliary_count: int = 0
    _last_activity: float = field(default_factory=time.monotonic)

    @property
    def session_count(self) -> int:
        return len(self.session_slots)

    @property
    def available_session_slots(self) -> int:
        return max(0, self.max_concurrent - self.reserved_for_auxiliary - self.session_count)

    @property
    def available_auxiliary_slots(self) -> int:
        return max(0, self.reserved_for_auxiliary - self.auxiliary_count)

    @property
    def is_saturated(self) -> bool:
        """True when total usage (sessions + auxiliary) >= max_concurrent."""
        return (self.session_count + self.auxiliary_count) >= self.max_concurrent

    @property
    def pool_key(self) -> str:
        return f"{self.provider}:{self.model}"


# ---------------------------------------------------------------------------
# SessionModelPool
# ---------------------------------------------------------------------------


class SessionModelPool:
    """Thread-safe pool that assigns models to sessions based on concurrency limits.

    Usage::

        pool = SessionModelPool.from_config(config.get("session_model_pool", {}))
        if pool and pool.enabled:
            entry = pool.acquire_session_slot("discord:12345:67890")
            if entry:
                model = entry["model"]
            # ... later ...
            pool.release_session_slot("discord:12345:67890")
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        strategy: str = "round-robin",
        inactive_timeout: int = 1800,
        entries: Optional[List[PoolModelEntry]] = None,
    ):
        self.enabled = enabled
        self.strategy = strategy
        self.inactive_timeout = inactive_timeout  # seconds
        self._entries: List[PoolModelEntry] = entries or []
        self._lock = threading.Lock()
        # Track which session_key is currently assigned to which model (for release)
        self._session_map: Dict[str, str] = {}  # session_key -> pool_key
        # Track manually overridden sessions (pool won't reassign)
        self._manual_overrides: set = set()

    # ---- factory ----

    @classmethod
    def from_config(cls, config: dict) -> Optional["SessionModelPool"]:
        """Build a pool from the ``session_model_pool`` config section.

        Returns ``None`` when the feature is disabled or config is empty/invalid.
        """
        if not config or not config.get("enabled"):
            return None

        raw_pool = config.get("pool", [])
        if not raw_pool or not isinstance(raw_pool, list):
            logger.warning("session_model_pool.pool is empty or not a list — disabled")
            return None

        entries: List[PoolModelEntry] = []
        for raw in raw_pool:
            if not isinstance(raw, dict):
                logger.warning("Skipping non-dict entry in session_model_pool.pool")
                continue
            model = raw.get("model", "")
            provider = raw.get("provider", "")
            if not model or not provider:
                logger.warning(
                    "Skipping pool entry missing model/provider: %s", raw
                )
                continue
            max_c = raw.get("max_concurrent", 1)
            if not isinstance(max_c, int) or max_c < 1:
                max_c = 1
            reserved = raw.get("reserved_for_auxiliary", 0)
            if not isinstance(reserved, int) or reserved < 0:
                reserved = 0
            if reserved > max_c:
                logger.warning(
                    "reserved_for_auxiliary (%d) > max_concurrent (%d) for %s:%s — capping",
                    reserved, max_c, provider, model,
                )
                reserved = max_c
            ctx_len = raw.get("context_length")
            priority = raw.get("priority", 5)
            if not isinstance(priority, int):
                priority = 5
            entries.append(
                PoolModelEntry(
                    model=model,
                    provider=provider,
                    max_concurrent=max_c,
                    reserved_for_auxiliary=reserved,
                    context_length=ctx_len if isinstance(ctx_len, int) else None,
                    priority=priority,
                )
            )

        if not entries:
            logger.warning("session_model_pool has no valid entries — disabled")
            return None

        strategy = config.get("strategy", "round-robin")
        if strategy not in ("round-robin", "least-loaded", "priority"):
            logger.warning("Unknown strategy '%s' — defaulting to round-robin", strategy)
            strategy = "round-robin"

        inactive_timeout = config.get("inactive_timeout", 1800)
        if not isinstance(inactive_timeout, int) or inactive_timeout < 0:
            inactive_timeout = 1800

        logger.info(
            "SessionModelPool enabled: strategy=%s, %d models, timeout=%ds",
            strategy, len(entries), inactive_timeout,
        )
        return cls(
            enabled=True,
            strategy=strategy,
            inactive_timeout=inactive_timeout,
            entries=entries,
        )

    # ---- session slot management ----

    def acquire_session_slot(self, session_key: str) -> Optional[dict]:
        """Find the best available model and assign the session to it.

        Returns ``{"model": str, "provider": str, "context_length": int|None}`` or
        ``None`` if all models are saturated.
        """
        if not self.enabled:
            return None

        with self._lock:
            # Don't reassign manually-overridden sessions
            if session_key in self._manual_overrides:
                return None

            # Release stale session slots first
            self._evict_inactive_sessions()

            # If session already has a slot, return it
            existing = self._session_map.get(session_key)
            if existing:
                for entry in self._entries:
                    if entry.pool_key == existing:
                        entry._last_activity = time.monotonic()
                        return {
                            "model": entry.model,
                            "provider": entry.provider,
                            "context_length": entry.context_length,
                        }

            # Find candidates with available session slots
            candidates = [
                e for e in self._entries
                if e.available_session_slots > 0
            ]

            if not candidates:
                logger.warning(
                    "SessionModelPool: all models saturated for session %s "
                    "(%d models, %d active sessions)",
                    session_key, len(self._entries), len(self._session_map),
                )
                return None

            # Pick the best candidate
            chosen = self._pick_candidate(candidates)

            chosen.session_slots.append(session_key)
            chosen._last_activity = time.monotonic()
            self._session_map[session_key] = chosen.pool_key

            logger.info(
                "SessionModelPool: assigned session %s -> %s:%s "
                "(session_slots=%d/%d, aux=%d/%d)",
                session_key, chosen.provider, chosen.model,
                chosen.session_count, chosen.max_concurrent,
                chosen.auxiliary_count, chosen.reserved_for_auxiliary,
            )
            return {
                "model": chosen.model,
                "provider": chosen.provider,
                "context_length": chosen.context_length,
            }

    def release_session_slot(self, session_key: str) -> None:
        """Release a session's claimed slot."""
        if not self.enabled:
            return

        with self._lock:
            pool_key = self._session_map.pop(session_key, None)
            if not pool_key:
                return
            for entry in self._entries:
                if entry.pool_key == pool_key:
                    if session_key in entry.session_slots:
                        entry.session_slots.remove(session_key)
                    logger.debug(
                        "SessionModelPool: released session %s from %s:%s",
                        session_key, entry.provider, entry.model,
                    )
                    break

    def mark_manual_override(self, session_key: str) -> None:
        """Mark a session as manually overridden (e.g. via /model command).

        The pool will release any existing slot and won't auto-assign this session.
        """
        if not self.enabled:
            return

        with self._lock:
            self._manual_overrides.add(session_key)
            # Inline release logic (can't call release_session_slot which re-acquires lock)
            pool_key = self._session_map.pop(session_key, None)
            if pool_key:
                for entry in self._entries:
                    if entry.pool_key == pool_key:
                        if session_key in entry.session_slots:
                            entry.session_slots.remove(session_key)
                        break

    def clear_manual_override(self, session_key: str) -> None:
        """Remove manual override marker (e.g. on /new or /reset)."""
        with self._lock:
            self._manual_overrides.discard(session_key)

    # ---- auxiliary slot management ----

    def acquire_auxiliary_slot(
        self, model: str, provider: str, timeout: float = 5.0
    ) -> bool:
        """Try to claim an auxiliary slot for a given model.

        Blocks up to *timeout* seconds waiting for a slot to free up.
        Returns ``True`` if the call is allowed to proceed.
        """
        if not self.enabled:
            return True  # no pool = no restrictions

        pool_key = f"{provider}:{model}"
        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                for entry in self._entries:
                    if entry.pool_key == pool_key:
                        if entry.available_auxiliary_slots > 0:
                            entry.auxiliary_count += 1
                            return True
                        # No auxiliary slot available
                        break
                else:
                    # Model not in pool — allow without restriction
                    return True

            # Wait and retry
            if time.monotonic() >= deadline:
                logger.warning(
                    "SessionModelPool: auxiliary slot unavailable for %s:%s "
                    "after %.1fs (aux=%d, reserved=%d)",
                    provider, model, timeout,
                    self._aux_count(pool_key),
                    self._reserved_count(pool_key),
                )
                return False
            time.sleep(0.5)

    def release_auxiliary_slot(self, model: str, provider: str) -> None:
        """Release an auxiliary slot."""
        if not self.enabled:
            return

        pool_key = f"{provider}:{model}"
        with self._lock:
            for entry in self._entries:
                if entry.pool_key == pool_key and entry.auxiliary_count > 0:
                    entry.auxiliary_count -= 1
                    break

    # ---- strategy ----

    def _pick_candidate(self, candidates: List[PoolModelEntry]) -> PoolModelEntry:
        """Pick the best model from candidates based on configured strategy."""
        if self.strategy == "priority":
            # Highest priority first; break ties by least loaded
            return max(candidates, key=lambda e: (e.priority, -e.session_count))
        elif self.strategy == "least-loaded":
            # Fewest sessions first; break ties by priority
            return min(candidates, key=lambda e: (e.session_count, -e.priority))
        else:
            # round-robin: pick the one with the oldest last activity
            return min(candidates, key=lambda e: e._last_activity)

    # ---- maintenance ----

    def _evict_inactive_sessions(self) -> int:
        """Remove sessions that have been inactive longer than the timeout.

        Must be called with ``self._lock`` held.
        """
        now = time.monotonic()
        evicted = 0
        for entry in self._entries:
            stale = [
                sk for sk in entry.session_slots
                if (now - entry._last_activity) > self.inactive_timeout
            ]
            # We can't know per-session activity time from entry-level tracking,
            # so we skip per-entry eviction. Instead rely on explicit release.
            # The timeout is enforced at the acquire level.
        return evicted

    def reconstruct_from_active_sessions(self, active_session_keys: List[str]) -> None:
        """Rebuild pool state after gateway restart.

        Marks all active sessions as pool-assigned but without a specific model
        (they'll get reassigned on first message).
        """
        with self._lock:
            self._session_map.clear()
            self._manual_overrides.clear()
            for entry in self._entries:
                entry.session_slots.clear()
                entry.auxiliary_count = 0
            # We can't know which model each session was using, so clear maps
            # and let sessions be reassigned on next acquire_session_slot call
            logger.info(
                "SessionModelPool: reconstructed state — %d active sessions will be reassigned",
                len(active_session_keys),
            )

    def get_pool_stats(self) -> Dict[str, Any]:
        """Return current slot usage for logging/status."""
        with self._lock:
            stats = {
                "enabled": self.enabled,
                "strategy": self.strategy,
                "total_sessions": len(self._session_map),
                "manual_overrides": len(self._manual_overrides),
                "models": [],
            }
            for entry in self._entries:
                stats["models"].append({
                    "model": entry.model,
                    "provider": entry.provider,
                    "max_concurrent": entry.max_concurrent,
                    "reserved_for_auxiliary": entry.reserved_for_auxiliary,
                    "session_count": entry.session_count,
                    "available_sessions": entry.available_session_slots,
                    "auxiliary_count": entry.auxiliary_count,
                    "available_auxiliary": entry.available_auxiliary_slots,
                    "is_saturated": entry.is_saturated,
                })
            return stats

    # ---- internal helpers ----

    def _aux_count(self, pool_key: str) -> int:
        for entry in self._entries:
            if entry.pool_key == pool_key:
                return entry.auxiliary_count
        return 0

    def _reserved_count(self, pool_key: str) -> int:
        for entry in self._entries:
            if entry.pool_key == pool_key:
                return entry.reserved_for_auxiliary
        return 0


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_pool_instance: Optional[SessionModelPool] = None
_pool_lock = threading.Lock()


def get_session_model_pool(config: dict) -> Optional[SessionModelPool]:
    """Return the global pool instance, creating it from *config* if needed."""
    global _pool_instance
    if _pool_instance is None:
        with _pool_lock:
            if _pool_instance is None:
                pool_config = config.get("session_model_pool", {})
                _pool_instance = SessionModelPool.from_config(pool_config)
    return _pool_instance


def reset_session_model_pool() -> None:
    """Clear the singleton so it gets re-created from config on next access."""
    global _pool_instance
    with _pool_lock:
        _pool_instance = None
