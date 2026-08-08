"""Circuit breaker with per-error-type cooldowns and manual reset.

State file: ~/.hermes/cache/web-search-circuit.json

Cooldown tiers (seconds):
  TRANSIENT  = 60   (timeout, 503, 502, connection reset)
  RATE_LIMIT = 120  (HTTP 429)
  QUOTA      = 86400 (credit/quota exhausted)
  AUTH_ERR   = 86400 (401, 403 — invalid key)
  UNKNOWN    = 300   (fallback)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def _default_state_file() -> str:
    """Return the default circuit breaker state file path under HERMES_HOME."""
    hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return os.path.join(hermes_home, "cache", "web-search-circuit.json")

# Default cooldowns per error category (seconds)
DEFAULT_COOLDOWNS = {
    "transient": 60,
    "rate_limit": 120,
    "quota": 86400,
    "auth": 86400,
    "unknown": 300,
}


class CircuitBreaker:
    """Per-backend circuit breaker with error-type-aware cooldowns.

    States per backend:
      CLOSED  — healthy, requests flow through
      OPEN    — in cooldown, backend skipped
    """

    def __init__(
        self,
        state_file: Optional[str] = None,
        cooldowns: Optional[Dict[str, int]] = None,
    ):
        self._path = Path(state_file or _default_state_file())
        self._cooldowns = {**DEFAULT_COOLDOWNS, **(cooldowns or {})}
        self._state: Dict[str, Any] = {}
        self._load()

    # ── public API ──

    def is_available(self, name: str) -> bool:
        """Check if backend is available (CLOSED or cooldown expired)."""
        entry = self._state.get(name)
        if not entry:
            return True
        if entry.get("state") == "open":
            expires = entry.get("open_until", 0)
            if time.time() < expires:
                return False
            # Cooldown expired — auto-close
            self._close(name)
            return True
        return True

    def get_available_backends(self, backends: List[str]) -> List[str]:
        """Filter backend list to only available ones."""
        return [b for b in backends if self.is_available(b)]

    def record_failure(self, name: str, error_type: str) -> int:
        """Record a failure. Returns cooldown seconds applied."""
        now = time.time()
        cooldown = self._cooldowns.get(error_type, self._cooldowns["unknown"])
        entry = self._state.get(name, {"failures": 0, "consecutive": 0})
        entry["failures"] = entry.get("failures", 0) + 1
        entry["consecutive"] = entry.get("consecutive", 0) + 1
        entry["last_error"] = error_type
        entry["last_failure"] = now
        entry["state"] = "open"
        entry["open_until"] = now + cooldown
        self._state[name] = entry
        self._save()
        logger.warning(
            "\u26a0 %s OPEN [%s] %ds (failures=%d)",
            name, error_type, cooldown, entry["failures"],
        )
        return cooldown

    def record_success(self, name: str, latency_ms: float = 0) -> None:
        """Record a success — close the circuit."""
        self._close(name)
        logger.debug("✓ %s CLOSED (latency=%.0fms)", name, latency_ms)

    def reset_backend(self, name: str) -> bool:
        """Manual reset: close circuit and clear failure stats for a backend.

        Returns True if backend existed, False if not found.
        """
        if name not in self._state:
            return False
        old_state = self._state[name].get("state", "closed")
        self._close(name)
        logger.info("↻ %s manually reset (was %s)", name, old_state)
        return True

    def get_state(self, name: str) -> Dict[str, Any]:
        """Get detailed state for a backend."""
        entry = self._state.get(name, {})
        now = time.time()
        if entry.get("state") == "open":
            remaining = max(0, entry.get("open_until", 0) - now)
            entry["cooldown_remaining_s"] = int(remaining)
        return entry

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states for all tracked backends."""
        now = time.time()
        result = {}
        for name, entry in self._state.items():
            if entry.get("state") == "open":
                remaining = max(0, entry.get("open_until", 0) - now)
                entry["cooldown_remaining_s"] = int(remaining)
            result[name] = entry
        return result

    def get_cooldowns(self) -> Dict[str, int]:
        """Return current cooldown config."""
        return dict(self._cooldowns)

    # ── internals ──

    def _close(self, name: str) -> None:
        entry = self._state.get(name, {})
        entry["state"] = "closed"
        entry["consecutive"] = 0
        entry.pop("open_until", None)
        self._state[name] = entry
        self._save()

    def _load(self) -> None:
        try:
            if self._path.exists():
                self._state = json.loads(self._path.read_text())
        except Exception as exc:
            logger.warning("Circuit breaker load failed: %s", exc)
            self._state = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._state, indent=2, ensure_ascii=False))
            tmp.replace(self._path)
        except Exception as exc:
            logger.warning("Circuit breaker save failed: %s", exc)
