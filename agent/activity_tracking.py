"""Turn-liveness activity tracking for ``AIAgent`` (gateway watchdog + session activity persistence).

``_touch_activity`` is the single write path; persistence is rate-limited and never raises.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import logging
import os
import threading
import time
from contextlib import suppress
from typing import Optional

from agent.session_activity import ActivityProvenance

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


def _activity_lock(obj) -> "threading.Lock":
    """Lazy per-instance ``_turn_liveness_activity_lock`` (so ``__new__``/SimpleNamespace doubles work)."""
    _lock = getattr(obj, "_turn_liveness_activity_lock", None)
    if _lock is None:
        _lock = threading.Lock()
        obj._turn_liveness_activity_lock = _lock
    return _lock


class ActivityTrackingMixin:
    """Liveness timestamps/labels and rate-limited session activity persistence."""

    def _liveness_activity_lock(self) -> "threading.Lock":
        """Shared lock for the activity clock and its generation counter.

        ``_touch_activity`` stamps under it and the liveness watchdog samples/commits under it, so a stall
        observation can never abort a turn that resumed in between.

        Created lazily so ``AIAgent.__new__``-based test doubles keep working. See #95663.
        """
        return _activity_lock(self)

    def _touch_activity(
        self, desc: str, *, provenance: Optional[ActivityProvenance] = None,
        force_persist: bool = False,
    ) -> None:
        """Update the last-activity timestamp and description (thread-safe).

        Bumps a monotonic generation under the activity lock so the watchdog can bind a stall observation to
        the exact ``(generation, timestamp)`` it sampled. Also bridges (rate-limited, best-effort) to the
        kanban heartbeat when this is a dispatcher-spawned worker, and to the durable SessionDB activity
        projection. ``provenance`` names special writers (compression); ``force_persist`` bypasses the
        SessionDB rate limit. Module-level lock helper, not ``self._liveness_activity_lock()``: doubles bind
        only ``_touch_activity`` (tests/run_agent/test_session_activity_persist.py).

        Bridge is rate-limited (60s) and best-effort — it never raises into the agent loop. See #31752.
        See #72016, #72039.
        """
        from agent.session_activity import (
            bound_activity_description, normalize_activity_provenance,
            reset_session_activity_persist_window,
        )

        with _activity_lock(self):
            self._turn_liveness_activity_generation = (
                getattr(self, "_turn_liveness_activity_generation", 0) + 1
            )
            self._last_activity_ts = time.time()
            self._last_activity_desc = bound_activity_description(desc)
            self._last_activity_provenance = normalize_activity_provenance(provenance)
            # Real progress invalidates a reserved abort claim; an in-flight watchdog interrupt must abandon
            # itself at the final mutation edge.
            self._turn_liveness_abort_claim = None
        if os.environ.get("HERMES_KANBAN_TASK"):
            # Never let the bridge break the loop; this guard covers import-time failures.
            with suppress(Exception):
                from tools.kanban_tools import (
                    heartbeat_current_worker_from_env, inject_new_comments_from_env
                )
                heartbeat_current_worker_from_env()
                # Fold new operator notes into the running turn (OUT-OF-BAND steer).
                inject_new_comments_from_env(self)
        if force_persist:
            reset_session_activity_persist_window(self)
        self._persist_session_activity_if_due()

    def _persist_session_activity_if_due(self) -> None:
        """Best-effort durable activity heartbeat for SessionDB consumers.

        Cadence pinned by ``SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS`` (config-independent). Fail-open:
        a failed write never raises into the agent loop.
        """
        session_id = getattr(self, "session_id", None)
        session_db = getattr(self, "_session_db", None)
        if not session_id or session_db is None:
            return
        touch = getattr(session_db, "touch_session_activity", None)
        if not callable(touch):
            return
        from agent.session_activity import (
            SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS, normalize_activity_provenance
        )

        now_mono = time.monotonic()
        last_mono = getattr(self, "_session_activity_last_persist_mono", 0.0)
        if (now_mono - last_mono) < SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS:
            return
        self._session_activity_last_persist_mono = now_mono
        try:
            touch(
                session_id,
                getattr(self, "_last_activity_ts", None),
                description=getattr(self, "_last_activity_desc", None),
                provenance=normalize_activity_provenance(
                    getattr(self, "_last_activity_provenance", None)
                ),
            )
        except Exception:
            # Heartbeat is observation-only; never let its I/O break the loop.
            logger.debug("session activity heartbeat write failed (ignored)", exc_info=True)

    def _reset_activity_labels_after_turn(self) -> None:
        """Drop mid-turn activity labels once the turn is no longer running.

        Keeps ``_last_activity_ts`` so idle/watchdog clocks stay continuous across turns; clears description +
        provenance so idle agents / SessionDB listings stop advertising the last mid-turn stamp.

        See #15654, #72039.
        """
        self._last_activity_desc = ""
        self._last_activity_provenance = ActivityProvenance.UNKNOWN
        session_id = getattr(self, "session_id", None)
        session_db = getattr(self, "_session_db", None)
        if not session_id or session_db is None:
            return
        clear = getattr(session_db, "clear_session_activity_labels", None)
        if not callable(clear):
            return
        with suppress(Exception):  # never let durable cleanup I/O break turn teardown
            clear(session_id)
