"""Provider cooldown / circuit breaker for Hermes Agent.

Tracks provider failures and applies escalating backoff to prevent
hammering providers that are rate-limiting, overloaded, or rejecting
requests with auth/billing errors.

Cooldown schedules:
- Transient errors (rate_limit, overloaded): 30s → 60s → 5min
- Permanent errors (auth_permanent, billing): 5min → 10min → 30min
- Generic auth errors: treated as transient (may be refreshable)

Thread-safe via threading.Lock for use in concurrent subagent scenarios.

This module is intentionally standalone — it does NOT depend on
provider_errors.py (P0) and uses plain string reason codes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


# ── Reason codes (plain strings, no enum dependency) ──────────────
# These match the categories that run_agent.py error handlers detect.
# When P0 (structured error classification) lands, a follow-up can
# map ErrorCategory → reason string.
REASON_RATE_LIMIT = "rate_limit"
REASON_AUTH = "auth"
REASON_AUTH_PERMANENT = "auth_permanent"
REASON_OVERLOADED = "overloaded"
REASON_BILLING = "billing"

# Reasons that use the "permanent" (longer) backoff schedule
_PERMANENT_REASONS = frozenset({REASON_AUTH_PERMANENT, REASON_BILLING})

# ── Backoff schedules (seconds) ──────────────────────────────────
# Index by min(error_count, len(schedule)) - 1
_TRANSIENT_BACKOFF = (30, 60, 300)       # 30s → 60s → 5min
_PERMANENT_BACKOFF = (300, 600, 1800)    # 5min → 10min → 30min


@dataclass
class ProviderHealthStats:
    """Runtime health counters for a provider endpoint."""

    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    last_error_reason: Optional[str] = None
    last_error_at: Optional[float] = None
    last_success_at: Optional[float] = None

    @property
    def total_calls(self) -> int:
        return self.success_count + self.error_count

    @property
    def error_rate(self) -> float:
        """Error rate as 0.0-1.0."""
        if self.total_calls == 0:
            return 0.0
        return self.error_count / self.total_calls

    @property
    def avg_latency_ms(self) -> float:
        if self.success_count == 0:
            return 0.0
        return self.total_latency_ms / self.success_count


@dataclass
class CooldownEntry:
    """Tracks cooldown state for a single provider endpoint."""

    provider_key: str        # 'provider::base_url'
    reason: str              # last failure reason
    error_count: int = 0     # consecutive failures
    cooldown_until: float = 0.0   # time.time() when cooldown expires
    last_failure_at: float = 0.0  # time.time() of most recent failure


def _backoff_seconds(error_count: int, reason: str) -> float:
    """Return the backoff duration for the given error count and reason."""
    schedule = _PERMANENT_BACKOFF if reason in _PERMANENT_REASONS else _TRANSIENT_BACKOFF
    idx = min(error_count, len(schedule)) - 1
    idx = max(idx, 0)
    return float(schedule[idx])


def _make_key(provider: str, base_url: str) -> str:
    """Build a canonical key for provider + endpoint."""
    return f"{provider}::{base_url or ''}"


class ProviderCooldownTracker:
    """Process-scoped tracker for provider cooldown / circuit breaker.

    Usage::

        tracker = get_cooldown_tracker()
        tracker.record_failure("openrouter", "https://openrouter.ai/api/v1", "rate_limit")
        entry = tracker.is_in_cooldown("openrouter", "https://openrouter.ai/api/v1")
        if entry:
            print(f"In cooldown until {entry.cooldown_until}")

    Thread-safe — all mutations are guarded by an internal lock.
    """

    _instance: Optional["ProviderCooldownTracker"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._cooldowns: Dict[str, CooldownEntry] = {}
        self._health: Dict[str, ProviderHealthStats] = {}
        self._lock = threading.Lock()

    # ── Singleton accessor ────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "ProviderCooldownTracker":
        """Return the process-scoped singleton, creating it on first call."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def _reset_singleton(cls) -> None:
        """Reset the singleton — for tests only."""
        with cls._instance_lock:
            cls._instance = None

    # ── Public API ────────────────────────────────────────────────

    def record_failure(self, provider: str, base_url: str, reason: str) -> CooldownEntry:
        """Record a failure and compute the new cooldown window.

        Returns the updated CooldownEntry.
        """
        key = _make_key(provider, base_url)
        now = time.time()

        with self._lock:
            entry = self._cooldowns.get(key)
            if entry is None:
                entry = CooldownEntry(provider_key=key, reason=reason)
                self._cooldowns[key] = entry

            entry.error_count += 1
            entry.reason = reason
            entry.last_failure_at = now

            backoff = _backoff_seconds(entry.error_count, reason)
            entry.cooldown_until = now + backoff

            # Update health stats
            health = self._health.get(key)
            if health is None:
                health = ProviderHealthStats()
                self._health[key] = health
            health.error_count += 1
            health.last_error_reason = reason
            health.last_error_at = now

            return entry

    def record_success(self, provider: str, base_url: str, latency_ms: Optional[float] = None) -> None:
        """Record a successful request — resets (closes) the circuit.

        Also updates health stats.  Pass *latency_ms* to track API latency.
        """
        key = _make_key(provider, base_url)
        now = time.time()
        with self._lock:
            self._cooldowns.pop(key, None)

            # Update health stats
            health = self._health.get(key)
            if health is None:
                health = ProviderHealthStats()
                self._health[key] = health
            health.success_count += 1
            health.last_success_at = now
            if latency_ms is not None:
                health.total_latency_ms += latency_ms

    def is_in_cooldown(self, provider: str, base_url: str) -> Optional[CooldownEntry]:
        """Check if a provider is currently in cooldown.

        Returns the CooldownEntry if still active, or None if OK.
        Automatically clears expired entries.
        """
        key = _make_key(provider, base_url)
        now = time.time()

        with self._lock:
            entry = self._cooldowns.get(key)
            if entry is None:
                return None

            if now >= entry.cooldown_until:
                # Cooldown expired — auto-clear
                del self._cooldowns[key]
                return None

            return entry

    def clear_all(self) -> None:
        """Reset all cooldown entries and health stats (for tests / session reset)."""
        with self._lock:
            self._cooldowns.clear()
            self._health.clear()

    def clear_provider(self, provider: str, base_url: str) -> None:
        """Clear cooldown for a specific provider endpoint."""
        key = _make_key(provider, base_url)
        with self._lock:
            self._cooldowns.pop(key, None)

    def get_cooldown_summary(self) -> dict:
        """Return a diagnostic summary of all active cooldowns.

        Returns a dict mapping provider keys to their status::

            {
                "openrouter::https://openrouter.ai/api/v1": {
                    "reason": "rate_limit",
                    "error_count": 2,
                    "remaining_seconds": 45.3,
                    "cooldown_until": 1700000045.3,
                    "last_failure_at": 1700000000.0,
                },
                ...
            }
        """
        now = time.time()
        summary = {}

        with self._lock:
            # Clean expired entries while building summary
            expired_keys = []
            for key, entry in self._cooldowns.items():
                if now >= entry.cooldown_until:
                    expired_keys.append(key)
                else:
                    summary[key] = {
                        "reason": entry.reason,
                        "error_count": entry.error_count,
                        "remaining_seconds": round(entry.cooldown_until - now, 1),
                        "cooldown_until": entry.cooldown_until,
                        "last_failure_at": entry.last_failure_at,
                    }
            for key in expired_keys:
                del self._cooldowns[key]

        return summary

    def get_health_stats(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, ProviderHealthStats]:
        """Return health stats, optionally filtered by provider/base_url.

        If *provider* and *base_url* are both given, return just that key.
        If only *provider* is given, return all keys whose provider segment matches.
        If neither is given, return everything.
        """
        with self._lock:
            if provider is not None and base_url is not None:
                key = _make_key(provider, base_url)
                entry = self._health.get(key)
                if entry is not None:
                    return {key: entry}
                return {}

            if provider is not None:
                prefix = f"{provider}::"
                return {k: v for k, v in self._health.items() if k.startswith(prefix)}

            # Return a shallow copy so callers can iterate safely
            return dict(self._health)

    def get_health_summary(self) -> dict:
        """Compact health summary suitable for /status display.

        Returns::

            {
                "openrouter::https://...": {
                    "total_calls": 15,
                    "success": 12,
                    "errors": 3,
                    "error_rate": 0.2,
                    "avg_latency_ms": 1234.5,
                    "last_error_reason": "rate_limit",
                },
                ...
            }
        """
        with self._lock:
            summary: dict = {}
            for key, stats in self._health.items():
                summary[key] = {
                    "total_calls": stats.total_calls,
                    "success": stats.success_count,
                    "errors": stats.error_count,
                    "error_rate": round(stats.error_rate, 3),
                    "avg_latency_ms": round(stats.avg_latency_ms, 1),
                    "last_error_reason": stats.last_error_reason,
                }
            return summary


def get_cooldown_tracker() -> ProviderCooldownTracker:
    """Module-level convenience accessor for the singleton tracker."""
    return ProviderCooldownTracker.get_instance()
