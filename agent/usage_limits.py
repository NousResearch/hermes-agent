"""Opt-in cooperative conversation-loop budgets (zero disables each field).

Checked only between outer iterations, not during calls, retries or tools, so a crossing
operation completes before the next boundary stops the loop: these are soft limits, not
hard deadlines or financial caps. The timer starts after turn preflight. Token and cost
limits observe this agent's session counters (reported usage plus folded subagent cost;
reset by /new, and a fresh process resuming a session starts from zero). Cost uses the
same estimate as /usage, so an unpriced model never trips ``session_cost_usd``. Limits are
re-read from config.yaml at each turn start, so raising one and sending "continue" resumes
a stopped session. The app-server runtime is not covered.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

_RAISE_HINT = ("Raise or clear `usage_limits.{key}` (`hermes config set usage_limits.{key} <value>`, "
               "0 disables) and send \"continue\" to pick up where it stopped.")


@dataclass(frozen=True)
class UsageLimitsConfig:
    """Ceilings from the ``usage_limits`` config.yaml section. 0 = disabled."""

    turn_wall_clock_seconds: int = 0
    turn_total_tokens: int = 0
    session_total_tokens: int = 0
    session_cost_usd: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> UsageLimitsConfig:
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            turn_wall_clock_seconds=_limit(data.get("turn_wall_clock_seconds")),
            turn_total_tokens=_limit(data.get("turn_total_tokens")),
            session_total_tokens=_limit(data.get("session_total_tokens")),
            session_cost_usd=_cost_limit(data.get("session_cost_usd")),
        )

    @property
    def enabled(self) -> bool:
        return bool(
            self.turn_wall_clock_seconds
            or self.turn_total_tokens
            or self.session_total_tokens
            or self.session_cost_usd
        )


def current_usage_limits() -> UsageLimitsConfig:
    """Limits from the active profile's config.yaml (mtime-cached read)."""
    from hermes_cli.config import load_config_readonly
    return UsageLimitsConfig.from_mapping(load_config_readonly().get("usage_limits"))


@dataclass(frozen=True)
class UsageBreach:
    """One exceeded ceiling: a stable code plus a user-facing explanation."""

    code: str
    message: str


class TurnUsageTracker:
    """Per-turn observer over wall-clock time and the session token/cost aggregates.

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

    def check(self, session_total_tokens: int, session_cost_usd: float = 0.0) -> UsageBreach | None:
        cfg = self.config
        if not cfg.enabled:
            return None

        if cfg.turn_wall_clock_seconds:
            elapsed = self._clock() - self._started
            if elapsed >= cfg.turn_wall_clock_seconds:
                return _breach("usage_limit_wall_clock", "turn_wall_clock_seconds",
                               f"Turn time limit reached ({int(elapsed)}s of {cfg.turn_wall_clock_seconds}s).")

        session_tokens = max(0, int(session_total_tokens or 0))
        if cfg.turn_total_tokens:
            turn_tokens = max(0, session_tokens - self._token_watermark)
            if turn_tokens >= cfg.turn_total_tokens:
                return _breach("usage_limit_turn_tokens", "turn_total_tokens",
                               f"Turn token limit reached ({turn_tokens:,} of {cfg.turn_total_tokens:,}).")

        if cfg.session_total_tokens and session_tokens >= cfg.session_total_tokens:
            return _breach("usage_limit_session_tokens", "session_total_tokens",
                           f"Session token limit reached ({session_tokens:,} of {cfg.session_total_tokens:,}).")

        spent = float(session_cost_usd or 0.0)
        if cfg.session_cost_usd and math.isfinite(spent) and spent >= cfg.session_cost_usd:
            return _breach("usage_limit_session_cost", "session_cost_usd",
                           f"Session cost limit reached (~{_usd(spent)} of {_usd(cfg.session_cost_usd)}).")

        return None


def _breach(code: str, key: str, headline: str) -> UsageBreach:
    return UsageBreach(code=code, message=(
        f"⏹️  {headline} Stopped before the next model call; completed work is kept. "
        + _RAISE_HINT.format(key=key)))


def _usd(amount: float) -> str:
    return f"${amount:.2f}" if amount >= 0.01 else f"${amount:.6f}"


def _limit(value: Any) -> int:
    """Parse an integer ceiling: positive int, anything else (or 0) disables."""
    if value is None or isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, parsed)


def _cost_limit(value: Any) -> float:
    """Parse a USD ceiling: positive finite number, anything else (or 0) disables."""
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return parsed if math.isfinite(parsed) and parsed > 0 else 0.0


__all__ = ["TurnUsageTracker", "UsageBreach", "UsageLimitsConfig", "current_usage_limits"]
