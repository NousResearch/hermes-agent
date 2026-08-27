"""Turn liveness watchdog (#95548): force-abort turns that stall silently.

A conversation turn can stall mid-flight (observed in #95548 between
"model returned tool_calls" and tool execution, after a slow model
response + desktop WS disconnect) with no error logged, no further
progress, and the durable session turn lease kept renewing — so nothing
ever force-aborts the turn and the session stays stuck until the process
is killed.

This module owns the watchdog policy end to end:

* configuration — :func:`resolve_turn_liveness_settings` reads the
  ``agent.turn_liveness`` section of config.yaml and validates it;
* state machine — :class:`TurnLivenessWatchdog` samples the agent's
  activity clock and drives the stall decision;
* thread mechanics — the polling loop, stop handling, and the stall
  commit point.

``AIAgent.run_conversation`` (``run_agent.py``) keeps only the smallest
integration seam: it resolves the settings, supplies the commit and
deactivate callbacks that own turn-lease state, and starts the thread
next to the durable lease refresher.

Config surface (config.yaml)::

    agent:
      turn_liveness:
        timeout_s: 600.0   # idle bound; <= 0 disables the watchdog
        poll_s: 15.0       # sampling interval (seconds)

Both values are validated. A non-numeric typo, ``NaN``, or ``Inf`` logs a
warning and falls back to the documented default — it never crashes
startup, and a bogus value can never silently disable the watchdog
(``NaN``) or freeze the watcher thread (``Inf`` poll).

Race safety (#95663 review): the watchdog samples the activity clock and
binds the abort decision to the observed ``(generation, timestamp)``
pair. The commit callback revalidates that pair under the *same* lock
``AIAgent._touch_activity`` uses to stamp the clock, so a turn that
resumed while the stall was being logged/emitted is never hard-cancelled
— it continues and its lease keeps renewing. Round-3 carried the
revalidated generation into ``AIAgent.interrupt``
(``require_generation``). Round-4 makes that generation an actual
cancellation claim consumed at the final mutation edge: ``interrupt``
reserves the claim under the activity lock, ``_touch_activity``
invalidates the reservation the instant real progress lands, and the
claim survives every blocking boundary, including the compression
commit fence. Round-6 consumes the claim and publishes the first
observable interrupt state in ONE activity-lock critical section, so a
turn that resumes can only ever interleave before that section (the
reservation is invalidated and the abort declines) or after it (the
interrupt already committed under the lock) — never between "claim
consumed" and "state published". A turn that resumes anywhere in the
window is never hard-cancelled, and an exceptional interrupt path
declines the abort fail-closed instead of mutating interrupt state.
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
    """One observation of the activity clock, bound to an abort decision.

    ``generation`` + ``activity_ts`` uniquely identify the observed stamp.
    The commit callback must revalidate this pair under the lock shared
    with ``AIAgent._touch_activity``; if it no longer matches, the
    observation is stale and the abort must be declined.
    """

    generation: int
    activity_ts: Optional[float]
    idle_seconds: float


def _warn_invalid_value(key: str, raw: Any, default: float) -> None:
    logger.warning(
        "Invalid %s in config.yaml: %r — falling back to default %.1f.",
        key,
        raw,
        default,
    )


def _resolve_finite_seconds(raw: Any, *, default: float, key: str) -> float:
    """Coerce one duration knob, rejecting typos, NaN and Inf.

    A non-numeric value must not raise into durable-turn startup, and a
    non-finite value must not silently change behavior (``NaN`` would
    disable the timeout via the ``> 0`` comparison; ``Inf`` would freeze
    the poll loop in ``Event.wait``).
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        _warn_invalid_value(key, raw, default)
        return default
    if not math.isfinite(value):
        _warn_invalid_value(key, raw, default)
        return default
    return value


def resolve_turn_liveness_settings(
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], float]:
    """Resolve ``(timeout_s, poll_s)`` from the ``agent.turn_liveness`` section.

    Precedence: explicit config.yaml value wins over the default; any
    invalid value (typo, NaN, Inf, non-positive poll) falls back to the
    default with a warning. ``timeout_s <= 0`` is the documented opt-out
    and yields ``(None, poll_s)`` — the caller then never arms the
    watchdog. The resolver never raises.
    """
    section: Dict[str, Any] = {}
    if isinstance(config, dict):
        agent_cfg = config.get("agent")
        if isinstance(agent_cfg, dict):
            raw_section = agent_cfg.get("turn_liveness")
            if isinstance(raw_section, dict):
                section = raw_section
            elif raw_section is not None:
                _warn_invalid_value(
                    "agent.turn_liveness", raw_section, DEFAULT_TURN_LIVENESS_TIMEOUT_S
                )

    timeout_s = _resolve_finite_seconds(
        section.get("timeout_s", DEFAULT_TURN_LIVENESS_TIMEOUT_S),
        default=DEFAULT_TURN_LIVENESS_TIMEOUT_S,
        key=_CONFIG_TIMEOUT_KEY,
    )
    poll_s = _resolve_finite_seconds(
        section.get("poll_s", DEFAULT_TURN_LIVENESS_POLL_S),
        default=DEFAULT_TURN_LIVENESS_POLL_S,
        key=_CONFIG_POLL_KEY,
    )
    if poll_s <= 0:
        _warn_invalid_value(_CONFIG_POLL_KEY, poll_s, DEFAULT_TURN_LIVENESS_POLL_S)
        poll_s = DEFAULT_TURN_LIVENESS_POLL_S
    # <= 0 is the documented opt-out, not an error.
    if timeout_s <= 0:
        timeout_s = None
    return timeout_s, poll_s


class TurnLivenessWatchdog:
    """Sampled-idle watchdog thread bound to one conversation turn.

    ``run_agent.py`` owns the turn-lease state (stop event, turn-active
    flag, interrupt plumbing); this class only reads the activity clock
    and calls back at the stall commit point. All synchronization with
    ``AIAgent._touch_activity`` goes through ``activity_lock``, which
    must be the SAME lock the agent stamps its activity clock with.
    """

    def __init__(
        self,
        agent: Any,
        *,
        session_id: str,
        timeout_s: float,
        poll_s: float,
        stop_event: threading.Event,
        activity_lock: threading.Lock,
        is_turn_active: Callable[[], bool],
        commit_abort: Callable[[ActivitySnapshot, str], bool],
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

    def make_thread(self) -> threading.Thread:
        """Build the (not yet started) watcher thread.

        ``run_agent.py`` creates the watchdog before the turn begins but
        starts the thread at turn entry, right after the turn-active flag
        and the activity clock are stamped.
        """
        return threading.Thread(
            target=self._watch,
            name="turn-liveness-watchdog",
            daemon=True,
        )

    def start(self) -> threading.Thread:
        """Spawn the watcher thread and return it (already running)."""
        thread = self.make_thread()
        thread.start()
        return thread

    def _watch(self) -> None:
        while not self._stop_event.wait(self._poll_s):
            snapshot = self._sample()
            if snapshot is None:
                # Turn is no longer active; nothing to watch.
                return
            if snapshot.idle_seconds < self._timeout_s:
                continue
            self._surface_stall(snapshot)
            # Commit point: bind the abort to the sampled generation/ts
            # and revalidate under the lock shared with `_touch_activity`.
            # If progress resumed while the stall was being surfaced, the
            # turn continues and this loop resumes sampling — the lease
            # keeps renewing. The commit also carries the revalidated
            # generation into the interrupt path, which reserves it as a
            # claim, survives every blocking boundary (compression
            # fence), and consumes it at the final mutation edge
            # immediately before publication (#95663 round-4) — progress
            # landing anywhere in that window declines the abort.
            if not self._commit_abort(snapshot, self._abort_message(snapshot)):
                continue
            # Stop renewing the durable lease: a wedge the hard interrupt
            # cannot unwind must not keep the lease alive forever (the
            # issue's "lease keeps renewing" masking). The TTL expiry then
            # lets stale-turn cleanup reclaim the row.
            self._deactivate_turn()
            return

    def _sample(self) -> Optional[ActivitySnapshot]:
        with self._activity_lock:
            if not self._is_turn_active():
                return None
            generation = getattr(
                self._agent, "_turn_liveness_activity_generation", 0
            )
            activity_ts = getattr(self._agent, "_last_activity_ts", None)
        now = time.time()
        if activity_ts is None:
            idle_seconds = 0.0
        else:
            idle_seconds = max(0.0, now - activity_ts)
        return ActivitySnapshot(
            generation=generation,
            activity_ts=activity_ts,
            idle_seconds=idle_seconds,
        )

    def _abort_message(self, snapshot: ActivitySnapshot) -> str:
        return (
            f"Turn made no progress for {int(snapshot.idle_seconds)}s; "
            "aborting to release the session."
        )

    def _surface_stall(self, snapshot: ActivitySnapshot) -> None:
        """Log the stall loudly and emit a UI-visible warning.

        The lock is deliberately NOT held here: acting on the observation
        happens at the commit point, after this surface window, where the
        observed generation/timestamp is revalidated.
        """
        session_id = getattr(self._agent, "session_id", None) or self._session_id
        last_desc = getattr(self._agent, "_last_activity_desc", None)
        logger.error(
            "Turn liveness watchdog fired for session %s: "
            "no progress for %.1fs (last activity: %r). "
            "Force-aborting the turn and stopping lease renewal (#95548).",
            session_id,
            snapshot.idle_seconds,
            last_desc,
        )
        emit_warning = getattr(self._agent, "_emit_warning", None)
        if not callable(emit_warning):
            return
        try:
            emit_warning(
                "⚠️ This turn stopped making progress "
                f"({int(snapshot.idle_seconds)}s without activity); "
                "aborting it so the session can recover."
            )
        except Exception:
            logger.debug("Failed to emit turn liveness warning", exc_info=True)
