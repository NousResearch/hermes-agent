"""
CLI Bridge Rate Tracker — Per-CLI rate limit and daily quota tracking.

"Every request counts. Especially the free ones." - The Collective

Tracks:
- Daily request counter with midnight reset
- Per-minute sliding window
- Cooldown tracking when rate limited
- Shared singleton across all bridges
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, Optional

from loguru import logger


@dataclass
class CLIRateState:
    """Rate state for a single CLI."""
    daily_count: int = 0
    daily_limit: Optional[int] = None
    daily_reset_date: str = ""
    minute_timestamps: list = field(default_factory=list)
    minute_limit: int = 30
    cooldown_until: float = 0.0
    last_request: float = 0.0


class RateTracker:
    """
    Singleton rate tracker for all CLI bridges.

    Tracks per-CLI daily quotas and per-minute sliding windows.
    Thread-safe for use across async tasks.
    """

    _instance: Optional["RateTracker"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._states: Dict[str, CLIRateState] = {}
                    cls._instance._state_lock = threading.Lock()
        return cls._instance

    def register_cli(
        self,
        cli_name: str,
        daily_limit: Optional[int] = None,
        minute_limit: int = 30,
    ):
        """Register a CLI with its rate limits."""
        with self._state_lock:
            if cli_name not in self._states:
                self._states[cli_name] = CLIRateState(
                    daily_limit=daily_limit,
                    minute_limit=minute_limit,
                    daily_reset_date=str(date.today()),
                )

    def can_request(self, cli_name: str) -> bool:
        """Check if a request is allowed for this CLI."""
        with self._state_lock:
            state = self._states.get(cli_name)
            if not state:
                return True

            now = time.time()

            # Check cooldown
            if now < state.cooldown_until:
                return False

            # Check daily limit
            today = str(date.today())
            if state.daily_reset_date != today:
                state.daily_count = 0
                state.daily_reset_date = today

            if state.daily_limit and state.daily_count >= state.daily_limit:
                logger.warning(f"[{cli_name}] Daily limit reached: {state.daily_count}/{state.daily_limit}")
                return False

            # Check per-minute window
            cutoff = now - 60
            state.minute_timestamps = [t for t in state.minute_timestamps if t > cutoff]
            if len(state.minute_timestamps) >= state.minute_limit:
                return False

            return True

    def record_request(self, cli_name: str):
        """Record that a request was made."""
        with self._state_lock:
            state = self._states.get(cli_name)
            if not state:
                return

            now = time.time()
            state.daily_count += 1
            state.minute_timestamps.append(now)
            state.last_request = now

    def record_rate_limit(self, cli_name: str, cooldown_seconds: float = 60.0):
        """Record that the CLI returned a rate limit error."""
        with self._state_lock:
            state = self._states.get(cli_name)
            if not state:
                return

            state.cooldown_until = time.time() + cooldown_seconds
            logger.warning(f"[{cli_name}] Rate limited, cooldown for {cooldown_seconds}s")

    def get_stats(self, cli_name: str) -> Dict:
        """Get rate stats for a CLI."""
        with self._state_lock:
            state = self._states.get(cli_name)
            if not state:
                return {"cli_name": cli_name, "registered": False}

            now = time.time()
            cutoff = now - 60
            recent = [t for t in state.minute_timestamps if t > cutoff]

            return {
                "cli_name": cli_name,
                "registered": True,
                "daily_count": state.daily_count,
                "daily_limit": state.daily_limit,
                "daily_remaining": (state.daily_limit - state.daily_count) if state.daily_limit else None,
                "minute_count": len(recent),
                "minute_limit": state.minute_limit,
                "in_cooldown": now < state.cooldown_until,
                "cooldown_remaining": max(0, state.cooldown_until - now),
            }

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get rate stats for all CLIs."""
        with self._state_lock:
            return {name: self.get_stats(name) for name in self._states}


def get_rate_tracker() -> RateTracker:
    """Get the singleton rate tracker."""
    return RateTracker()
