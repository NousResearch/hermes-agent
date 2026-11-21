"""Operator-facing wait notice for long provider silences (#92550).

The 30s heartbeat is gateway liveness, not a warning. Once a request has been
silent past the notice threshold this module decides WHAT the status line says
(neutral: which phase of the wait we are in, which watchdog would reconnect and
when) and WHETHER to rewrite it: once when the silence starts, again only when
the wait phase changes or a watchdog deadline is near. Watchdog thresholds and
retry policy live with the watchdogs; this is presentation only.
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

NEAR_DEADLINE_SECS = 15.0


def _near_deadline(watchdog: Optional[tuple[str, float]]) -> bool:
    return watchdog is not None and watchdog[1] <= NEAR_DEADLINE_SECS


_PHASE_TEXT = {
    # Codex Responses (non-stream request path)
    "first_event": "{n}s waiting for the first provider event",
    "reconnect": "{n}s waiting for the first provider event after reconnect",
    "pre_progress": "provider stream open; {n}s without substantive model progress",
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

    The status line is transient, so each silence is ALSO written to the app log
    exactly twice — ``provider-wait start`` when the notice first appears and
    ``provider-wait end`` with the outcome when the wait closes. Two lines per
    silence, never one per heartbeat: enough to count and alert on long provider
    waits after the fact, without turning agent.log into a drumbeat.
    """

    def __init__(self) -> None:
        self.model: Optional[str] = None
        self.provider: Optional[str] = None
        self._waited_secs = 0.0
        self._logged_start = False
        self.reset()

    def reset(self, outcome: str = "resumed") -> None:
        self._log_end(outcome)
        self.phase: Optional[str] = None
        self.watchdog_label: Optional[str] = None
        self.near_shown = False

    def _log_end(self, outcome: str) -> None:
        if not self._logged_start:
            return
        self._logged_start = False
        logger.info("provider-wait end: model=%s provider=%s phase=%s waited=%.0fs outcome=%s",
                    self.model or "unknown", self.provider or "unknown", self.phase or "unknown",
                    self._waited_secs, outcome)

    def should_emit(self, phase: str, watchdog: Optional[tuple[str, float]], *,
                    model: Optional[str] = None, provider: Optional[str] = None,
                    silence_secs: Optional[float] = None) -> bool:
        if model is not None:
            self.model = model
        if provider is not None:
            self.provider = provider
        if silence_secs is not None:
            self._waited_secs = float(silence_secs)
        label = watchdog[0] if watchdog is not None else None
        near = _near_deadline(watchdog)
        emit = self.phase != phase or self.watchdog_label != label or (near and not self.near_shown)
        self.phase, self.watchdog_label = phase, label
        if near:
            self.near_shown = True
        if emit and not self._logged_start:
            self._logged_start = True
            logger.info("provider-wait start: model=%s provider=%s phase=%s waited=%.0fs watchdog=%s",
                        self.model or "unknown", self.provider or "unknown", phase,
                        self._waited_secs, label or "none")
        return emit
