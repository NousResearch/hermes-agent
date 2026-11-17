"""Codex app-server plan limits: the windows codex pushes, and what a usage-limit failure says.

``codex app-server`` sends ``account/rateLimits/updated`` after every Responses call (it reads the
``x-codex-*`` headers itself), and a turn the plan refuses fails with
``codexErrorInfo: usageLimitExceeded`` and a message that only names a wall-clock time
("try again at 7:33 AM"). :class:`CodexRateLimitState` folds the sparse updates into one view,
lands it in the account-usage cache (the model picker gauge and ``/usage`` read it without an HTTP
call of their own) and turns a usage-limit failure into a reset time the user and the fallback
cooldown can both use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from agent.account_usage import (
    _CODEX_WINDOW_LABELS_BY_SECONDS,
    _CODEX_WINDOW_POSITIONAL_LABELS,
    AccountUsageSnapshot,
    _format_reset,
    _snapshot,
    _title_case_slug,
    _usage_windows,
)
from agent.account_usage_cache import remember_account_usage

APP_SERVER_USAGE_SOURCE = "app_server"
USAGE_LIMIT_ERROR_INFO = "usageLimitExceeded"
# The metered bucket the subscription quota lives in; other limit ids (per-model buckets) are not
# the account's session/weekly windows.
_ACCOUNT_LIMIT_IDS = frozenset({None, "codex"})
_WINDOW_KEYS = ("primary", "secondary")
_EXHAUSTED_PERCENT = 100.0


@dataclass
class CodexRateLimitState:
    """Latest plan windows one codex app-server session reported, plus this turn's failure kind."""

    windows: dict[str, dict] = field(default_factory=dict)
    plan: Optional[str] = None
    turn_error_info: Any = None

    def merge(self, params: dict) -> Optional[AccountUsageSnapshot]:
        """Fold one ``account/rateLimits/updated`` into the state; the snapshot to cache, if any.

        Updates are sparse: a null window or plan means "not in this update", never "cleared"."""
        rate_limits = params.get("rateLimits")
        if not isinstance(rate_limits, dict) or rate_limits.get("limitId") not in _ACCOUNT_LIMIT_IDS:
            return None
        for key in _WINDOW_KEYS:
            window = rate_limits.get(key)
            if isinstance(window, dict) and window.get("usedPercent") is not None:
                self.windows[key] = dict(window)
        plan = rate_limits.get("planType")
        if plan and plan != "unknown":
            self.plan = _title_case_slug(plan)
        return self.snapshot()

    def snapshot(self) -> Optional[AccountUsageSnapshot]:
        windows = _usage_windows(self.windows, self._labels(), "usedPercent", "resetsAt")
        if not windows:
            return None
        return _snapshot("openai-codex", APP_SERVER_USAGE_SOURCE, windows, [], plan=self.plan)

    def _labels(self) -> tuple[tuple[str, str], ...]:
        """Session/Weekly by the window's duration, else by position (same rule as the usage API)."""
        labels = []
        for key, (_, fallback) in zip(_WINDOW_KEYS, _CODEX_WINDOW_POSITIONAL_LABELS):
            minutes = (self.windows.get(key) or {}).get("windowDurationMins")
            label = _CODEX_WINDOW_LABELS_BY_SECONDS.get(int(minutes) * 60, fallback) if isinstance(
                minutes, (int, float)) else fallback
            labels.append((key, label))
        return tuple(labels)

    def exhausted_reset_at(self) -> Optional[datetime]:
        """When the account is whole again: the LATEST reset among its spent windows."""
        snapshot = self.snapshot()
        resets = [w.reset_at for w in (snapshot.windows if snapshot else ())
                  if w.reset_at is not None and (w.used_percent or 0) >= _EXHAUSTED_PERCENT]
        return max(resets) if resets else None


def on_rate_limits_updated(state: CodexRateLimitState, params: dict) -> None:
    remember_account_usage("openai-codex", state.merge(params))


def on_error_notification(state: CodexRateLimitState, params: dict) -> None:
    """Remember why the turn failed; a retried error (``willRetry``) is not the turn's outcome."""
    error = params.get("error")
    if isinstance(error, dict) and not params.get("willRetry"):
        state.turn_error_info = error.get("codexErrorInfo")


def usage_limit_reset(state: Optional[CodexRateLimitState], turn: Any) -> Optional[datetime]:
    """The reset time of a turn the plan refused, or None (other failure, or no window reported)."""
    if state is None or turn.error is None or state.turn_error_info != USAGE_LIMIT_ERROR_INFO:
        return None
    return state.exhausted_reset_at()


def usage_limit_message(reset_at: datetime) -> str:
    return (f"Codex usage limit reached — it resets {_format_reset(reset_at)}. "
            "Run /usage to see your windows.")
