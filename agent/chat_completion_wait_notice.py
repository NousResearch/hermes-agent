"""Operator-facing wait notice for long provider silences (#92550).

The 30s heartbeat is gateway liveness, not a warning. Once a request has been
silent past the notice threshold this module decides WHAT the status line says
(neutral: which phase of the wait we are in, which watchdog would reconnect and
when) and WHETHER to rewrite it: once when the silence starts, again only when
the wait phase changes or a watchdog deadline is near. Watchdog thresholds and
retry policy live with the watchdogs; this is presentation only.
"""

import math
from typing import Optional

from agent.i18n import tl

NEAR_DEADLINE_SECS = 15.0


def _near_deadline(watchdog: Optional[tuple[str, float]]) -> bool:
    return watchdog is not None and watchdog[1] <= NEAR_DEADLINE_SECS


def wait_notice_text(model: str, silence_secs: float, phase: str,
                     watchdog: Optional[tuple[str, float]] = None) -> str:
    """One neutral status line. ``watchdog`` is ``(label, seconds_until_it_fires)``.

    Phases: Codex Responses ``first_event`` / ``reconnect`` / ``pre_progress`` / ``post_event``; chat-completions
    streaming ``first_chunk`` / ``post_chunk``. One catalog line per phase, plus ``_watchdog`` (a reconnect is
    scheduled) and ``_near`` (it fires within ``NEAR_DEADLINE_SECS``: "still waiting") variants."""
    if watchdog is None:
        return tl(f"display.wait.{phase}", model=model, n=int(silence_secs))
    label, remaining = watchdog
    variant = "near" if _near_deadline(watchdog) else "watchdog"
    return tl(f"display.wait.{phase}_{variant}", model=model, n=int(silence_secs), watchdog=label,
              remaining=max(0, int(remaining)))


def codex_watchdog_deadline(*, stale_timeout: float, ttfb_enabled: bool, ttfb_timeout: float,
    last_event_ts: Optional[float], last_progress_ts: Optional[float],
    retry_started_ts: Optional[float], call_start: float, idle_enabled: bool,
    idle_timeout: float, idle_requires_progress: bool, elapsed: float,
    progress_timeout: float = 0.0) -> Optional[tuple[str, float]]:
    """Earliest enabled Codex watchdog as ``(label, seconds_until_it_fires)``; None when
    none applies (disabled/infinite, or its deadline already passed)."""
    deadlines: list[tuple[str, float]] = []
    if math.isfinite(stale_timeout):
        deadlines.append(("wall-clock stale", stale_timeout))
    attempt_offset = max(0.0, retry_started_ts - call_start) if retry_started_ts is not None else 0.0
    if last_event_ts is None:
        if ttfb_enabled and math.isfinite(ttfb_timeout):
            deadlines.append(("TTFB", attempt_offset + ttfb_timeout))
    elif progress_timeout > 0 and last_progress_ts is None:
        deadlines.append(("first progress", attempt_offset + progress_timeout))
    elif (not idle_requires_progress or last_progress_ts is not None) and idle_enabled and math.isfinite(idle_timeout):
        deadlines.append(("stream idle", max(0.0, last_event_ts - call_start) + idle_timeout))
    if not deadlines:
        return None
    label, deadline = min(deadlines, key=lambda d: d[1])
    if deadline <= elapsed:
        return None
    return label, deadline - elapsed


class WaitNoticeState:
    """Per-request memory of what the status line currently shows.

    ``should_emit`` is True for the first notice of a silence, for a phase or
    watchdog change, and once when the applicable deadline comes within
    ``NEAR_DEADLINE_SECS``; every other heartbeat only touches liveness.
    ``reset`` when activity resumes so the next silence gets a fresh notice.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.phase: Optional[str] = None
        self.watchdog_label: Optional[str] = None
        self.near_shown = False

    def should_emit(self, phase: str, watchdog: Optional[tuple[str, float]]) -> bool:
        label = watchdog[0] if watchdog is not None else None
        near = _near_deadline(watchdog)
        emit = self.phase != phase or self.watchdog_label != label or (near and not self.near_shown)
        self.phase, self.watchdog_label = phase, label
        if near:
            self.near_shown = True
        return emit
