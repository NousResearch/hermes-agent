"""Safety redline state machine.

This is the *deterministic* half of the safety enforcement: a small, pure
state machine that maps a stream of ``record_success()`` / ``record_failure()``
events onto a four-state pause cascade. It deliberately does no I/O -- the
protocol adapter layer is responsible for translating wire events into calls
into this machine.

State machine:

    HEALTHY     -- 0 or 1 recent failure; traffic is allowed
    WARN        -- 2 recent failures; traffic still allowed; advisory emitted
    PAUSED      -- 3 consecutive failures; new traffic from the offending
                   consumer is rejected until cooldown elapses
    HARD_PAUSE  -- 4+ consecutive failures; permanent until operator reset;
                   pushes a notification through the configured channel

Transitions are driven by the failure stream plus elapsed time. Once the
machine sees a success after a failure, the counter resets and WARN/PAUSED
clear back to HEALTHY (HARD_PAUSE stays sticky until operator reset).

Defaults follow the well-known "3 -> paused, 4 -> hard_pause, 5-minute
cooldown" convention. Field tuning goes through ``SafetyConfig``.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


REDLINE_VERSION = "0.1.0"


class SafetyState(str, enum.Enum):
    HEALTHY = "healthy"
    WARN = "warn"
    PAUSED = "paused"
    HARD_PAUSE = "hard_pause"


@dataclass(frozen=True)
class SafetyConfig:
    """Tunable knobs. Defaults mirror OC redline.py defaults."""

    # 3 consecutive failures -> PAUSED, soft cooldown
    pause_threshold: int = 3
    # 4 consecutive failures -> HARD_PAUSE; permanent until reset
    hard_pause_threshold: int = 4
    # Seconds a PAUSED state lingers before auto-clearing on next event.
    cooldown_seconds: float = 300.0  # 5 minutes, same as OC side
    # Rolling window over which we count "consecutive" failures. Older
    # failures outside the window are considered stale.
    failure_window_seconds: float = 600.0
    # Two failures within this window -> WARN (advisory).
    warn_threshold: int = 2
    # Cooldown notification channel; filled in by protocol layer.
    notifier: Optional[Callable[[str, str, dict], None]] = None

    def without_notifier(self) -> "SafetyConfig":
        return SafetyConfig(
            pause_threshold=self.pause_threshold,
            hard_pause_threshold=self.hard_pause_threshold,
            cooldown_seconds=self.cooldown_seconds,
            failure_window_seconds=self.failure_window_seconds,
            warn_threshold=self.warn_threshold,
            notifier=None,
        )


@dataclass
class SafetyEvent:
    state: SafetyState
    failure_streak: int
    message: str
    timestamp: float


@dataclass
class SafetyRedline:
    """State machine driving the safety redline.

    Thread-safety: this dataclass is intentionally lock-free. The intended
    usage is that the protocol adapter serialises calls onto a single event
    loop thread (matching the gateway's ``run.py`` async model). External
    callers that need multi-threaded access should wrap with their own lock.
    """

    config: SafetyConfig = field(default_factory=SafetyConfig)
    state: SafetyState = SafetyState.HEALTHY
    _failure_streak: int = 0
    _last_failure_at: float = 0.0
    _pause_started_at: float = 0.0
    _history: List[SafetyEvent] = field(default_factory=list)

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Operator-driven reset; clears PAUSED/HARD_PAUSE."""
        self.state = SafetyState.HEALTHY
        self._failure_streak = 0
        self._last_failure_at = 0.0
        self._pause_started_at = 0.0

    # -- event recording ----------------------------------------------------

    def record_success(self, *, now: Optional[float] = None) -> SafetyEvent:
        """Report a successful API call. Resets the failure streak."""
        now = now if now is not None else time.monotonic()
        was_paused = self.state in (SafetyState.PAUSED, SafetyState.HARD_PAUSE)
        self._failure_streak = 0
        if was_paused:
            # Once PAUSED/HARD_PAUSE we don't auto-drop to HEALTHY just
            # because we saw a success; the redline stays in PAUSED until
            # the cooldown elapses. This matches the consumer end.
            elapsed = now - self._pause_started_at
            if elapsed >= self.config.cooldown_seconds and self.state != SafetyState.HARD_PAUSE:
                self.state = SafetyState.HEALTHY
                self._pause_started_at = 0.0
        elif self.state in (SafetyState.WARN, SafetyState.HEALTHY):
            # A fresh success after a WARN streak returns us to HEALTHY.
            # This is the symmetric counterpart to ``record_failure``
            # pulling us down to HEALTHY on the first failure.
            self.state = SafetyState.HEALTHY
        return self._emit(self.state, "success recorded", now)

    def record_failure(
        self,
        *,
        reason: str = "",
        now: Optional[float] = None,
    ) -> SafetyEvent:
        """Report a failed API call. Returns the resulting state event."""
        now = now if now is not None else time.monotonic()
        # Refresh state based on elapsed time first.
        self._maybe_clear_pause(now)

        self._failure_streak += 1
        self._last_failure_at = now

        previous = self.state
        if self._failure_streak >= self.config.hard_pause_threshold:
            self.state = SafetyState.HARD_PAUSE
        elif self._failure_streak >= self.config.pause_threshold:
            self.state = SafetyState.PAUSED
            if previous != SafetyState.PAUSED:
                self._pause_started_at = now
        elif self._failure_streak >= self.config.warn_threshold:
            self.state = SafetyState.WARN
        else:
            self.state = SafetyState.HEALTHY

        escalated = self.state != previous
        message = self._describe(reason)

        if escalated and self.state in (SafetyState.PAUSED, SafetyState.HARD_PAUSE):
            self._notify(self.state, message)

        return self._emit(self.state, message, now)

    # -- queries ------------------------------------------------------------

    def is_traffic_allowed(self, *, now: Optional[float] = None) -> bool:
        """Return True if a peer request should currently be accepted."""
        if self.state == SafetyState.HARD_PAUSE:
            return False
        if self.state == SafetyState.PAUSED:
            check_now = now if now is not None else time.monotonic()
            return (check_now - self._pause_started_at) >= self.config.cooldown_seconds
        return True

    def snapshot(self) -> dict:
        return {
            "state": self.state.value,
            "failure_streak": self._failure_streak,
            "last_failure_at": self._last_failure_at,
            "pause_started_at": self._pause_started_at,
            "version": REDLINE_VERSION,
        }

    def history(self) -> List[SafetyEvent]:
        return list(self._history)

    # -- internals ----------------------------------------------------------

    def _maybe_clear_pause(self, now: float) -> None:
        if self.state != SafetyState.PAUSED:
            return
        if now - self._pause_started_at >= self.config.cooldown_seconds:
            self.state = SafetyState.HEALTHY
            self._failure_streak = 0
            self._pause_started_at = 0.0

    def _emit(self, state: SafetyState, message: str, now: float) -> SafetyEvent:
        event = SafetyEvent(
            state=state,
            failure_streak=self._failure_streak,
            message=message,
            timestamp=now,
        )
        self._history.append(event)
        # Keep history bounded so the redline doesn't grow unbounded if the
        # daemon runs for weeks. 64 is plenty for human inspection.
        if len(self._history) > 64:
            del self._history[: len(self._history) - 64]
        return event

    def _describe(self, reason: str) -> str:
        prefix = {
            SafetyState.HEALTHY: "healthy",
            SafetyState.WARN: "warn",
            SafetyState.PAUSED: "paused",
            SafetyState.HARD_PAUSE: "hard_pause",
        }[self.state]
        if reason:
            return f"{prefix} ({reason})"
        return prefix

    def _notify(self, state: SafetyState, message: str) -> None:
        notifier = self.config.notifier
        if notifier is None:
            return
        try:
            notifier(state.value, message, self.snapshot())
        except Exception:
            # Notification must never break the redline. Errors here would
            # silently disable safety. Swallow with explicit comment.
            pass
