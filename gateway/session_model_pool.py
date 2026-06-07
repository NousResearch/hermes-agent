"""Session model pool — concurrency-aware auto-assignment for multi-session gateways.

Distributes sessions and auxiliary calls across a configured pool of models,
respecting per-model concurrency limits and reserving slots for auxiliary tasks.

**Important:** The module-level singleton (`get_session_model_pool`) caches the
pool instance for the lifetime of the gateway process.  Runtime config changes
to ``session_model_pool`` require a gateway restart (or calling
``reset_session_model_pool()`` before the next access).
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
    session_slots: Dict[str, float] = field(default_factory=dict)  # session_key -> last_activity
    auxiliary_count: int = 0

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
        self._entries_by_key: Dict[str, PoolModelEntry] = {
            e.pool_key: e for e in self._entries
        }
        self._lock = threading.Lock()
        # Condition variable for auxiliary slot waiting (replaces polling).
        self._aux_condition = threading.Condition(self._lock)
        # Track which session_key is currently assigned to which model (for release).
        self._session_map: Dict[str, str] = {}  # session_key -> pool_key
        # Track manually overridden sessions (pool won't reassign).
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
            if priority < 1 or priority > 10:
                logger.warning(
                    "priority (%d) out of 1-10 range for %s:%s — clamping", priority, provider, model,
                )
                priority = max(1, min(priority, 10))
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

        # Deduplicate pool keys — if two entries share the same
        # provider:model, only the LAST entry is kept. Earlier duplicates
        # are logged and discarded to prevent inconsistent slot tracking.
        _deduped: Dict[str, PoolModelEntry] = {}
        for _entry in entries:
            _key = _entry.pool_key
            if _key in _deduped:
                logger.warning(
                    "session_model_pool: duplicate pool_key '%s' (%s and %s). "
                    "Keeping the last entry; discarding the earlier one.",
                    _key, _deduped[_key].model, _entry.model,
                )
            _deduped[_key] = _entry
        entries = list(_deduped.values())

        strategy = config.get("strategy", "round-robin")
        if strategy not in ("round-robin", "least-loaded", "priority"):
            logger.warning("Unknown strategy '%s' — defaulting to round-robin", strategy)
            strategy = "round-robin"

        inactive_timeout = config.get("inactive_timeout", 1800)
        if not isinstance(inactive_timeout, (int, float)) or inactive_timeout < 0:
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
            # Don't reassign manually-overridden sessions.
            if session_key in self._manual_overrides:
                return None

            # Evict stale sessions based on per-session activity timestamps.
            self._evict_inactive_sessions()

            # If session already has a slot, refresh its timestamp and return it.
            existing = self._session_map.get(session_key)
            if existing:
                entry = self._entries_by_key.get(existing)
                if entry:
                    entry.session_slots[session_key] = time.monotonic()
                    return {
                        "model": entry.model,
                        "provider": entry.provider,
                        "context_length": entry.context_length,
                    }

            # Find candidates with available session slots.
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

            # Pick the best candidate based on strategy.
            chosen = self._pick_candidate(candidates)

            chosen.session_slots[session_key] = time.monotonic()
            self._session_map[session_key] = chosen.pool_key

            logger.debug(
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
            entry = self._entries_by_key.get(pool_key)
            if entry:
                entry.session_slots.pop(session_key, None)
                logger.debug(
                    "SessionModelPool: released session %s from %s:%s",
                    session_key, entry.provider, entry.model,
                )

    def mark_manual_override(self, session_key: str) -> None:
        """Mark a session as manually overridden (e.g. via /model command).

        The pool will release any existing slot and won't auto-assign this session.
        """
        if not self.enabled:
            return

        with self._lock:
            self._manual_overrides.add(session_key)
            # Inline release (can't call release_session_slot which re-acquires lock).
            pool_key = self._session_map.pop(session_key, None)
            if pool_key:
                entry = self._entries_by_key.get(pool_key)
                if entry:
                    entry.session_slots.pop(session_key, None)

    def clear_manual_override(self, session_key: str) -> None:
        """Remove manual override marker (e.g. on /new or /reset)."""
        with self._lock:
            self._manual_overrides.discard(session_key)

    # ---- auxiliary slot management ----

    def acquire_auxiliary_slot(
        self, model: str, provider: str, timeout: float = 5.0
    ) -> bool:
        """Try to claim an auxiliary slot for a given model.

        Uses ``threading.Condition`` to wait efficiently (no busy-polling).
        Returns ``True`` if the call is allowed to proceed.
        """
        if not self.enabled:
            return True  # no pool = no restrictions

        pool_key = f"{provider}:{model}"
        deadline = time.monotonic() + timeout

        with self._aux_condition:
            while True:
                entry = self._entries_by_key.get(pool_key)
                if entry:
                    if entry.available_auxiliary_slots > 0:
                        entry.auxiliary_count += 1
                        return True
                    # No auxiliary slot available — wait for release.
                else:
                    # Model not in pool — allow without restriction.
                    return True

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "SessionModelPool: auxiliary slot unavailable for %s:%s "
                        "after %.1fs (aux=%d, reserved=%d)",
                        provider, model, timeout,
                        self._aux_count(pool_key),
                        self._reserved_count(pool_key),
                    )
                    return False

                # Wait for a slot to be released (signaled by release_auxiliary_slot).
                self._aux_condition.wait(timeout=min(remaining, 1.0))

    def release_auxiliary_slot(self, model: str, provider: str) -> None:
        """Release an auxiliary slot."""
        if not self.enabled:
            return

        pool_key = f"{provider}:{model}"
        with self._aux_condition:
            entry = self._entries_by_key.get(pool_key)
            if entry and entry.auxiliary_count > 0:
                entry.auxiliary_count -= 1
            # Wake up any threads waiting for auxiliary slots.
            self._aux_condition.notify_all()

    # ---- strategy ----

    def _pick_candidate(self, candidates: List[PoolModelEntry]) -> PoolModelEntry:
        """Pick the best model from candidates based on configured strategy."""
        if self.strategy == "priority":
            # Highest priority first; break ties by least loaded.
            return max(candidates, key=lambda e: (e.priority, -e.session_count))
        elif self.strategy == "least-loaded":
            # Fewest sessions first; break ties by priority.
            return min(candidates, key=lambda e: (e.session_count, -e.priority))
        else:
            # round-robin: pick the entry whose oldest session timestamp is earliest.
            def _oldest_activity(e: PoolModelEntry) -> float:
                if e.session_slots:
                    return min(e.session_slots.values())
                return 0.0
            return min(candidates, key=_oldest_activity)

    # ---- maintenance ----

    def _evict_inactive_sessions(self) -> int:
        """Remove sessions that have been inactive longer than the timeout.

        Must be called with ``self._lock`` held.
        """
        now = time.monotonic()
        evicted = 0
        for entry in self._entries:
            stale_keys = [
                sk for sk, ts in entry.session_slots.items()
                if (now - ts) > self.inactive_timeout
            ]
            for sk in stale_keys:
                entry.session_slots.pop(sk, None)
                self._session_map.pop(sk, None)
                evicted += 1
        if evicted:
            logger.info(
                "SessionModelPool: evicted %d inactive sessions", evicted
            )
        return evicted

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
        entry = self._entries_by_key.get(pool_key)
        return entry.auxiliary_count if entry else 0

    def _reserved_count(self, pool_key: str) -> int:
        entry = self._entries_by_key.get(pool_key)
        return entry.reserved_for_auxiliary if entry else 0


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_pool_instance: Optional[SessionModelPool] = None
_pool_lock = threading.Lock()


def get_session_model_pool(config: dict) -> Optional[SessionModelPool]:
    """Return the global pool instance, creating it from *config* if needed.

    The pool is cached for the lifetime of the gateway process.
    Runtime changes to ``session_model_pool`` in ``config.yaml`` require
    either a gateway restart or a call to ``reset_session_model_pool()``
    before the next access.
    """
    global _pool_instance
    if _pool_instance is None:
        with _pool_lock:
            if _pool_instance is None:
                pool_config = config.get("session_model_pool", {})
                _pool_instance = SessionModelPool.from_config(pool_config)
    return _pool_instance


def reset_session_model_pool() -> None:
    """Clear the singleton so it gets re-created from config on next access.

    Call this after runtime config changes to ``session_model_pool``.
    """
    global _pool_instance
    with _pool_lock:
        _pool_instance = None
