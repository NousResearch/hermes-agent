"""Cooldown tracking for provider failover."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class _CooldownState:
    """Internal per-key state."""
    count: int = 0
    until: float = 0.0   # monotonic timestamp when cooldown expires
    reason: str = "rate_limit"


def build_cooldown_key(provider: str, api_key: Optional[str], reason: str) -> str:
    """Build a provider- or provider:key-scoped cooldown key."""
    provider = (provider or "").strip().lower()
    if reason == "billing" or not api_key:
        return provider
    # Never persist or log a raw credential fragment. A digest keeps cooldowns
    # key-specific even when structured provider keys share a common prefix.
    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return f"{provider}:{fingerprint}"


def _is_safe_persisted_key(key: str) -> bool:
    """Return whether a persisted key contains no legacy credential fragment."""
    provider, separator, fingerprint = key.rpartition(":")
    if not separator:
        return True  # provider-scoped billing key
    return bool(provider) and len(fingerprint) == 16 and all(
        char in "0123456789abcdef" for char in fingerprint
    )


class CooldownManager:
    """Thread-safe cooldown tracker for provider keys."""

    def __init__(
        self,
        base_seconds: float = 60.0,
        multiplier: float = 5.0,
        max_seconds: float = 3600.0,
        billing_base_hours: float = 5.0,
        billing_max_hours: float = 24.0,
        storage_path: Union[Path, None, "Literal[False]"] = None,
    ) -> None:
        self._base_seconds = base_seconds
        self._multiplier = multiplier
        self._max_seconds = max_seconds
        self._billing_base_hours = billing_base_hours
        self._billing_max_hours = billing_max_hours
        self._states: Dict[str, _CooldownState] = {}
        self._lock = threading.Lock()

        if storage_path is False:
            self._storage_path: Optional[Path] = None
        elif storage_path is None:
            try:
                from hermes_constants import get_hermes_home
                self._storage_path = get_hermes_home() / "cooldowns.json"
            except Exception:
                self._storage_path = None
        else:
            self._storage_path = Path(storage_path)

        self._load()

    def is_cooling(self, key: str) -> bool:
        """Return True if *key* is currently on cooldown."""
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return False
            return time.monotonic() < state.until

    def mark_failure(
        self,
        key: str,
        reason: Literal["rate_limit", "billing"],
        cooldown_seconds: Optional[float] = None,
    ) -> float:
        """Record a failure and return the new cooldown in seconds."""
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = _CooldownState()
                self._states[key] = state
            if cooldown_seconds is None:
                state.count += 1
            state.reason = reason

            if cooldown_seconds is not None:
                cooldown_seconds = max(0.0, cooldown_seconds)
            elif reason == "billing":
                hours = min(
                    self._billing_base_hours * (2 ** (state.count - 1)),
                    self._billing_max_hours,
                )
                cooldown_seconds = hours * 3600.0
            else:  # rate_limit
                cooldown_seconds = min(
                    self._base_seconds * (self._multiplier ** (state.count - 1)),
                    self._max_seconds,
                )

            state.until = time.monotonic() + cooldown_seconds

        logger.info(
            "Cooldown: key=%r reason=%s count=%d duration=%.0fs",
            key, reason, state.count, cooldown_seconds,
        )
        self._persist()
        return cooldown_seconds

    def clear(self, key: str) -> None:
        """Remove cooldown state for *key* (e.g. after a successful call)."""
        with self._lock:
            removed = self._states.pop(key, None)
        if removed is not None:
            logger.debug("Cooldown cleared for key=%r", key)
        self._persist()

    def get_all_states(self) -> Dict[str, dict]:
        """Return a snapshot of all cooldown states."""
        now = time.monotonic()
        with self._lock:
            snapshot = {}
            for key, state in self._states.items():
                remaining = max(0.0, state.until - now)
                snapshot[key] = {
                    "count": state.count,
                    "until": state.until,
                    "cooling": now < state.until,
                    "remaining_seconds": remaining,
                }
        return snapshot

    def get_cooldown_status(self) -> dict:
        """Return a summary dict for logging or display."""
        states = self.get_all_states()
        cooling = [k for k, v in states.items() if v["cooling"]]
        expired = [k for k, v in states.items() if not v["cooling"] and v["count"] > 0]
        return {
            "total_tracked": len(states),
            "cooling": cooling,
            "expired": expired,
            "details": states,
        }

    def _load(self) -> None:
        """Load persisted cooldown state from disk.

        Converts wall-time ``until_wall`` values to monotonic offsets.
        Entries that have already expired are silently skipped.
        """
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            now_wall = time.time()
            now_mono = time.monotonic()
            discarded_legacy_key = False
            with self._lock:
                for key, entry in data.items():
                    if not _is_safe_persisted_key(key):
                        discarded_legacy_key = True
                        continue
                    until_wall = float(entry.get("until_wall", 0))
                    remaining = until_wall - now_wall
                    if remaining <= 0:
                        continue  # already expired — prune it
                    state = _CooldownState(
                        count=int(entry.get("count", 1)),
                        until=now_mono + remaining,
                        reason=str(entry.get("reason", "rate_limit")),
                    )
                    self._states[key] = state
            if discarded_legacy_key:
                self._persist()
        except Exception as exc:
            logger.warning("Failed to load cooldown state from %s: %s", self._storage_path, exc)

    def _persist(self) -> None:
        """Write active (non-expired) cooldown state to disk atomically."""
        if self._storage_path is None:
            return
        now_mono = time.monotonic()
        now_wall = time.time()
        try:
            with self._lock:
                data = {}
                for key, state in self._states.items():
                    remaining = state.until - now_mono
                    if remaining <= 0:
                        continue  # expired, omit
                    data[key] = {
                        "reason": state.reason,
                        "count": state.count,
                        "until_wall": now_wall + remaining,
                    }
            tmp_path = self._storage_path.with_suffix(".json.tmp")
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp_path, self._storage_path)
        except Exception as exc:
            logger.warning("Failed to persist cooldown state to %s: %s", self._storage_path, exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_singleton: Optional[CooldownManager] = None
_singleton_lock = threading.Lock()


def get_cooldown_manager() -> CooldownManager:
    """Return the process-wide CooldownManager singleton.

    The singleton is initialized with defaults on first call.  Tests may
    replace it via :func:`set_cooldown_manager` or simply instantiate their
    own :class:`CooldownManager` directly.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = CooldownManager()
    return _singleton


def set_cooldown_manager(manager: CooldownManager) -> None:
    """Replace the module-level singleton (useful for tests or custom config)."""
    global _singleton
    with _singleton_lock:
        _singleton = manager
