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

NEAR_DEADLINE_SECS = 15.0


def _near_deadline(watchdog: Optional[tuple[str, float]]) -> bool:
    return watchdog is not None and watchdog[1] <= NEAR_DEADLINE_SECS


_PHASE_TEXT = {
    # Codex Responses (non-stream request path)
    "first_event": "{n}s waiting for the first provider event",
    "reconnect": "{n}s waiting for the first provider event after reconnect",
    "post_event": "provider stream active; {n}s without stream events",
    # Chat-completions streaming path
    "first_chunk": "{n}s waiting for the first stream chunk",
    "post_chunk": "stream open; {n}s without stream output",
}


def wait_notice_text(model: str, silence_secs: float, phase: str,
                     watchdog: Optional[tuple[str, float]] = None) -> str:
    """One neutral status line. ``watchdog`` is ``(label, seconds_until_it_fires)``."""
    lead = "still waiting on" if _near_deadline(watchdog) else "waiting on"
    text = f"⏳ {lead} {model} — " + _PHASE_TEXT[phase].format(n=int(silence_secs))
    if watchdog is not None:
        label, remaining = watchdog
        text += f" (auto-reconnect: {label} watchdog in {max(0, int(remaining))}s)"
    return text


def codex_watchdog_deadline(*, stale_timeout: float, ttfb_enabled: bool, ttfb_timeout: float,
    last_event_ts: Optional[float], last_progress_ts: Optional[float],
    retry_started_ts: Optional[float], call_start: float, idle_enabled: bool,
    idle_timeout: float, idle_requires_progress: bool, elapsed: float) -> Optional[tuple[str, float]]:
    """Earliest enabled Codex watchdog as ``(label, seconds_until_it_fires)``; None when
    none applies (disabled/infinite, or its deadline already passed)."""
    deadlines: list[tuple[str, float]] = []
    if math.isfinite(stale_timeout):
        deadlines.append(("wall-clock stale", stale_timeout))
    if retry_started_ts is not None:
        if ttfb_enabled and math.isfinite(ttfb_timeout):
            deadlines.append(("TTFB", max(0.0, retry_started_ts - call_start) + ttfb_timeout))
    elif last_event_ts is None:
        if ttfb_enabled and math.isfinite(ttfb_timeout):
            deadlines.append(("TTFB", ttfb_timeout))
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
