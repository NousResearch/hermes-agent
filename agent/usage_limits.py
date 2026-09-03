"""Opt-in per-turn / per-session usage belts: wall-clock and token ceilings.

``IterationBudget`` bounds how many tool-loop passes a turn may take, but
nothing bounds how LONG a turn may run (slow local generations make 90
iterations an hours-long turn) or how many TOKENS a session may consume
(a runaway cloud session burns prepaid balance silently). These limits are
the missing belt for unattended 24/7 operation.

Design notes, mirroring ``agent.tool_guardrails``:

- Pure module: the tracker only observes and answers; the conversation loop
  owns breaking out of the turn.
- Token accounting taps the existing monotonic ``agent.session_total_tokens``
  aggregate (fed by every provider path). It only grows — there is no refund
  path — so limits inherit the retry-surviving property: an attempt that
  reported usage stays counted even if the call is later retried. Failed
  attempts that never returned a usage payload are NOT counted; the
  wall-clock ceiling covers stall-and-retry pathologies instead.
- Everything is disabled (0) by default: interactive sessions see no change
  unless config.yaml opts in via the ``usage_limits`` section.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class UsageLimitsConfig:
    """Ceilings from the ``usage_limits`` config.yaml section. 0 = disabled."""

    turn_wall_clock_seconds: int = 0
    turn_total_tokens: int = 0
    session_total_tokens: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "UsageLimitsConfig":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            turn_wall_clock_seconds=_limit(data.get("turn_wall_clock_seconds")),
            turn_total_tokens=_limit(data.get("turn_total_tokens")),
            session_total_tokens=_limit(data.get("session_total_tokens")),
        )

    @property
    def enabled(self) -> bool:
        return bool(
            self.turn_wall_clock_seconds
            or self.turn_total_tokens
            or self.session_total_tokens
        )


@dataclass(frozen=True)
class UsageBreach:
    """One exceeded ceiling: a stable code plus a user-facing explanation."""

    code: str
    message: str


class TurnUsageTracker:
    """Per-turn observer over wall-clock time and the session token aggregate.

    Instantiate at the head of a turn (captures the start instant and the
    session token watermark), then call :meth:`check` at each loop boundary.
    The first exceeded ceiling wins; ``None`` means carry on.
    """

    def __init__(
        self,
        config: UsageLimitsConfig | None,
        session_tokens_at_start: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or UsageLimitsConfig()
        self._clock = clock
        self._started = clock()
        self._token_watermark = max(0, int(session_tokens_at_start or 0))

    def check(self, session_total_tokens: int) -> UsageBreach | None:
        cfg = self.config
        if not cfg.enabled:
            return None

        if cfg.turn_wall_clock_seconds:
            elapsed = self._clock() - self._started
            if elapsed >= cfg.turn_wall_clock_seconds:
                return UsageBreach(
                    code="usage_limit_wall_clock",
                    message=(
                        f"Turn wall-clock limit reached ({int(elapsed)}s >= "
                        f"{cfg.turn_wall_clock_seconds}s). Wrapping up with "
                        "the work done so far."
                    ),
                )

        session_tokens = max(0, int(session_total_tokens or 0))
        if cfg.turn_total_tokens:
            turn_tokens = max(0, session_tokens - self._token_watermark)
            if turn_tokens >= cfg.turn_total_tokens:
                return UsageBreach(
                    code="usage_limit_turn_tokens",
                    message=(
                        f"Turn token limit reached ({turn_tokens} >= "
                        f"{cfg.turn_total_tokens}). Wrapping up with the "
                        "work done so far."
                    ),
                )

        if cfg.session_total_tokens and session_tokens >= cfg.session_total_tokens:
            return UsageBreach(
                code="usage_limit_session_tokens",
                message=(
                    f"Session token limit reached ({session_tokens} >= "
                    f"{cfg.session_total_tokens}). Wrapping up with the "
                    "work done so far."
                ),
            )

        return None


def _limit(value: Any) -> int:
    """Parse a ceiling: non-negative int, anything else (or 0) disables."""
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


__all__ = ["UsageLimitsConfig", "UsageBreach", "TurnUsageTracker"]
