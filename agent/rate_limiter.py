"""Per-model rate limit handler with stepped cooldown.

Tracks 429 / rate-limit errors per model and applies a stepped cooldown
ladder:

    1st hit  →  30 s
    2nd hit  →  60 s
    3rd+ hit → 300 s  (5 min)

The step counter resets automatically after 10 minutes of *no* rate-limit
hits for a given model.

Thread-safe: all mutable state is guarded by a single ``threading.Lock``.

Usage example (inside an API retry loop)::

    from agent.rate_limiter import rate_limiter

    # Before calling the API – honour any active cooldown
    remaining = rate_limiter.check_rate_limit(model)
    if remaining > 0:
        time.sleep(remaining)

    try:
        response = client.chat.completions.create(...)
    except RateLimitError:
        cooldown = rate_limiter.record_rate_limit(model)
        print(f"Rate limited on {model}, cooling down for {cooldown}s")
        time.sleep(cooldown)
        # … retry …
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stepped cooldown ladder (seconds)
_COOLDOWN_STEPS: tuple[int, ...] = (30, 60, 300)

# After this many seconds with no new rate-limit hits the step counter resets.
_RESET_WINDOW: float = 600.0  # 10 minutes


# ---------------------------------------------------------------------------
# Internal per-model state
# ---------------------------------------------------------------------------

@dataclass
class _ModelCooldownState:
    """Mutable cooldown state for a single model."""

    # How many consecutive rate-limit hits (1-indexed).
    step: int = 0

    # ``time.monotonic()`` timestamp when the current cooldown ends.
    cooldown_until: float = 0.0

    # ``time.monotonic()`` of the last hit – used for the reset window.
    last_hit: float = 0.0


# ---------------------------------------------------------------------------
# Public API – singleton ``RateLimiter``
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe, per-model rate-limit handler with stepped cooldown."""

    def __init__(
        self,
        cooldown_steps: tuple[int, ...] = _COOLDOWN_STEPS,
        reset_window: float = _RESET_WINDOW,
    ) -> None:
        self._cooldown_steps = cooldown_steps
        self._reset_window = reset_window
        self._lock = threading.Lock()
        self._models: Dict[str, _ModelCooldownState] = {}

    # -- helpers ----------------------------------------------------------

    def _get_state(self, model: str) -> _ModelCooldownState:
        """Return (or create) the state object for *model*.  Caller must hold ``_lock``."""
        if model not in self._models:
            self._models[model] = _ModelCooldownState()
        return self._models[model]

    def _maybe_reset(self, state: _ModelCooldownState, now: float) -> None:
        """Reset the step counter if the reset window has elapsed since the last hit.

        Caller must hold ``_lock``.
        """
        if state.last_hit and (now - state.last_hit) >= self._reset_window:
            state.step = 0

    # -- public interface -------------------------------------------------

    def check_rate_limit(self, model: str) -> float:
        """Return remaining cooldown seconds for *model*, or ``0`` if none."""
        now = time.monotonic()
        with self._lock:
            state = self._get_state(model)
            remaining = max(0.0, state.cooldown_until - now)
        return remaining

    def record_rate_limit(self, model: str) -> float:
        """Record a rate-limit hit for *model* and return the cooldown duration (seconds).

        The returned value is the number of seconds to wait before the next
        attempt.
        """
        now = time.monotonic()
        with self._lock:
            state = self._get_state(model)

            # Reset step counter if the reset window elapsed.
            self._maybe_reset(state, now)

            # Advance the step (clamped to the ladder length).
            state.step = min(state.step + 1, len(self._cooldown_steps))

            # Look up the cooldown for this step (1-indexed → 0-indexed).
            cooldown = self._cooldown_steps[state.step - 1]

            state.cooldown_until = now + cooldown
            state.last_hit = now

        return float(cooldown)

    def get_step(self, model: str) -> int:
        """Return the current step number for *model* (0 means no active cooldown)."""
        now = time.monotonic()
        with self._lock:
            state = self._get_state(model)
            self._maybe_reset(state, now)
            return state.step

    def get_cooldown_status(self) -> Dict[str, Dict[str, float]]:
        """Return a snapshot of all models with an active cooldown.

        Returns a dict mapping model name → ``{"remaining": <secs>, "step": <int>}``.
        Models whose cooldown has already expired are omitted.
        """
        now = time.monotonic()
        result: Dict[str, Dict[str, float]] = {}
        with self._lock:
            for model, state in self._models.items():
                remaining = max(0.0, state.cooldown_until - now)
                if remaining > 0:
                    result[model] = {
                        "remaining": round(remaining, 2),
                        "step": state.step,
                    }
        return result

    def reset(self, model: str | None = None) -> None:
        """Reset cooldown state.  If *model* is ``None``, reset everything."""
        with self._lock:
            if model is None:
                self._models.clear()
            elif model in self._models:
                del self._models[model]


# Module-level singleton for convenient import.
rate_limiter = RateLimiter()
