"""Turn liveness watchdog: force-abort turns that stall silently.

A turn can wedge mid-flight with no error and its durable lease still renewing, so nothing
ever frees the session. This module owns config resolution (``agent.turn_liveness``,
validated — a typo, NaN or Inf warns and falls back), the sampling state machine, and the
watcher thread. Race safety: the abort decision is bound to the observed
``(generation, timestamp)``; the commit callback revalidates it under the same lock
``_touch_activity`` stamps with, so a turn that resumed is never hard-cancelled.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TURN_LIVENESS_TIMEOUT_S = 600.0
DEFAULT_TURN_LIVENESS_POLL_S = 15.0
MIN_TURN_LIVENESS_POLL_S = 0.01

_CONFIG_TIMEOUT_KEY = "agent.turn_liveness.timeout_s"
_CONFIG_POLL_KEY = "agent.turn_liveness.poll_s"


class ActivitySnapshot(NamedTuple):
    """One activity-clock observation; ``(generation, activity_ts)`` must be
    revalidated by the commit callback under the shared lock."""

    generation: int
    activity_ts: Optional[float]
    idle_seconds: float


def _warn_invalid_value(key: str, raw: Any, default: float) -> None:
    logger.warning("Invalid %s in config.yaml: %r — falling back to default %.1f.", key, raw, default)


def _resolve_finite_seconds(raw: Any, *, default: float, key: str) -> float:
    """Coerce one duration knob, rejecting typos, NaN and Inf (never raises).

    NaN would silently disable the timeout via the ``> 0`` comparison and
    Inf would freeze the poll loop in ``Event.wait``.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        _warn_invalid_value(key, raw, default)
        return default
    return value


def resolve_turn_liveness_settings(
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], float]:
    """Resolve ``(timeout_s, poll_s)``; ``timeout_s <= 0`` opts out (``None``).

    Invalid values (typo, NaN, Inf, non-positive poll) warn and fall back to
    the default. Never raises.
    """
    agent_cfg = config.get("agent") if isinstance(config, dict) else None
    raw_section = agent_cfg.get("turn_liveness") if isinstance(agent_cfg, dict) else None
    section: Dict[str, Any] = raw_section if isinstance(raw_section, dict) else {}
    if raw_section is not None and not isinstance(raw_section, dict):
        _warn_invalid_value("agent.turn_liveness", raw_section, DEFAULT_TURN_LIVENESS_TIMEOUT_S)

    timeout_s = _resolve_finite_seconds(
        section.get("timeout_s", DEFAULT_TURN_LIVENESS_TIMEOUT_S),
        default=DEFAULT_TURN_LIVENESS_TIMEOUT_S, key=_CONFIG_TIMEOUT_KEY,
    )
    poll_s = _resolve_finite_seconds(
        section.get("poll_s", DEFAULT_TURN_LIVENESS_POLL_S), default=DEFAULT_TURN_LIVENESS_POLL_S,
        key=_CONFIG_POLL_KEY,
    )
    if poll_s <= 0:
        _warn_invalid_value(_CONFIG_POLL_KEY, poll_s, DEFAULT_TURN_LIVENESS_POLL_S)
        poll_s = DEFAULT_TURN_LIVENESS_POLL_S
    if timeout_s <= 0:
        timeout_s = None
    return timeout_s, poll_s


class TurnLivenessWatchdog:
    """Sampled-idle watchdog bound to one conversation turn (polls on the
    shared periodic scheduler thread).

    ``activity_lock`` must be the SAME lock ``AIAgent._touch_activity`` stamps
    the activity clock with; run_agent owns the lease state and callbacks.
    """

    def __init__(
        self, agent: Any, *, session_id: str, timeout_s: float, poll_s: float,
        stop_event: threading.Event, activity_lock: threading.Lock,
        is_turn_active: Callable[[], bool], commit_abort: Callable[[ActivitySnapshot, str], bool],
        deactivate_turn: Callable[[], None],
    ) -> None:
        self._agent = agent
        self._session_id = session_id
        self._timeout_s = float(timeout_s)
        self._poll_s = max(MIN_TURN_LIVENESS_POLL_S, float(poll_s))
        self._stop_event = stop_event
        self._activity_lock = activity_lock
        self._is_turn_active = is_turn_active
        self._commit_abort = commit_abort
        self._deactivate_turn = deactivate_turn

    def schedule(self):
        """Start polling on the shared periodic scheduler thread; returns the cancel handle.
        Scheduled at turn entry, after the turn-active flag and activity clock are stamped."""
        from agent.periodic_scheduler import schedule

        return schedule(self._tick, self._poll_s)

    def _tick(self):
        """One poll. Returns False when the watchdog is finished."""
        if self._stop_event.is_set():
            return False
        snapshot = self._sample()
        if snapshot is None:
            return False  # turn no longer active
        if snapshot.idle_seconds < self._timeout_s:
            return None
        # Observational only: the commit below can still veto the abort if progress
        # resumed; the definitive settlement is _surface_committed_abort.
        # Pre-commit surface is OBSERVATIONAL only: it reports the stall and that a recovery attempt is
        # beginning. It must not claim the abort or the lease withdrawal has committed — the next operation
        # can still veto the outcome. The definitive aborted/lease-stopped settlement is published by
        # _surface_committed_abort only after _commit_abort succeeds and the turn is deactivated (#95663
        # review).
        self._surface_stall(snapshot)
        message = f"Turn made no progress for {int(snapshot.idle_seconds)}s; aborting to release the session."
        if not self._commit_abort(snapshot, message):
            return None
        # Stop renewing the lease so a wedge the interrupt cannot unwind expires via TTL.
        self._deactivate_turn()
        self._surface_committed_abort(snapshot)
        return False

    def _sample(self) -> Optional[ActivitySnapshot]:
        with self._activity_lock:
            if not self._is_turn_active():
                return None
            generation = getattr(self._agent, "_turn_liveness_activity_generation", 0)
            activity_ts = getattr(self._agent, "_last_activity_ts", None)
        idle_seconds = 0.0 if activity_ts is None else max(0.0, time.time() - activity_ts)
        return ActivitySnapshot(generation, activity_ts, idle_seconds)

    def _emit_warning(self, text: str, debug_msg: str) -> None:
        emit_warning = getattr(self._agent, "_emit_warning", None)
        if not callable(emit_warning):
            return
        try:
            emit_warning(text)
        except Exception:
            logger.debug(debug_msg, exc_info=True)

    def _surface_stall(self, snapshot: ActivitySnapshot) -> None:
        """Log + UI-warn that recovery is beginning (not that it committed). Rate-limited per
        activity generation so repeatedly declined aborts do not re-log every poll."""
        generation = snapshot.generation
        if getattr(self, "_last_surfaced_generation", None) == generation:
            return
        self._last_surfaced_generation = generation
        logger.error(
            "Turn liveness watchdog fired for session %s: "
            "no progress for %.1fs (last activity: %r). "
            "Attempting recovery: force-interrupting the turn and "
            "stopping lease renewal if it cannot resume (#95548).",
            getattr(self._agent, "session_id", None) or self._session_id,
            snapshot.idle_seconds,
            getattr(self._agent, "_last_activity_desc", None),
        )
        self._emit_warning(
            "⚠️ This turn stopped making progress "
            f"({int(snapshot.idle_seconds)}s without activity); "
            "attempting recovery so the session can continue.",
            "Failed to emit turn liveness warning",
        )

    def _surface_committed_abort(self, snapshot: ActivitySnapshot) -> None:
        """Publish the definitive settlement once the abort has authority.

        Runs only once ``_commit_abort`` succeeded (the interrupt was published) and the turn lease was
        deactivated: the turn IS force-aborted and lease renewal IS stopped, so stating that is now true.
        Separated from the pre-commit surface so a declined abort never reports a committed outcome (#95663
        review).
        """
        logger.error(
            "Turn liveness watchdog aborted turn for session %s: "
            "no progress for %.1fs; turn interrupted and lease renewal "
            "stopped (#95548).",
            getattr(self._agent, "session_id", None) or self._session_id,
            snapshot.idle_seconds,
        )
        self._emit_warning(
            "⚠️ Turn aborted by the liveness watchdog "
            f"({int(snapshot.idle_seconds)}s without activity); "
            "lease renewal stopped so the session can be reclaimed. "
            "You can retry your message.",
            "Failed to emit committed-abort warning",
        )
