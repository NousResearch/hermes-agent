"""B-4: the no-progress guard must release the conversation, not only interrupt the turn.

A turn can wedge while its lease keeps being renewed, so the session stays owned by a turn that
will never finish: nothing else frees it and inbound deliveries are answered with a lease timeout
forever. The guard therefore has to do two things - hard-interrupt the turn, and stop renewing the
lease so the row expires and the next turn can take the conversation.
"""

from __future__ import annotations

import os
import threading
import time

from agent.turn_facade_lease import DurableTurnLease
from agent.turn_liveness import (
    DEFAULT_TURN_LIVENESS_TIMEOUT_S,
    TurnLivenessWatchdog,
)
from hermes_state import SessionDB


class _WedgeAgent:
    """A turn that is active but has made no model call and no tool completion."""

    def __init__(self, idle_seconds: float = 2000.0):
        self.session_id = "s1"
        self._turn_liveness_activity_generation = 7
        self._last_activity_ts = time.time() - idle_seconds
        self._last_activity_desc = "wedge"
        self.interrupts = []
        self.statuses = []
        self._lock = threading.Lock()

    def _liveness_activity_lock(self):
        return self._lock

    def _touch_activity(self, desc=""):
        pass

    def _emit_warning(self, text):
        self.statuses.append(text)

    def _emit_status(self, text):
        self.statuses.append(text)

    def interrupt(self, message, *, hard_cancel=False, require_generation=None):
        self.interrupts.append((message, hard_cancel, require_generation))
        return True


def _watchdog(agent, lease, timeout_s: float) -> TurnLivenessWatchdog:
    agent._lock = threading.Lock()
    return TurnLivenessWatchdog(
        agent,
        session_id="s1",
        timeout_s=timeout_s,
        poll_s=0.01,
        stop_event=lease.stop,
        activity_lock=agent._liveness_activity_lock(),
        is_turn_active=lease.is_turn_active,
        commit_abort=lease.commit_liveness_abort,
        deactivate_turn=lease.stop_refresher,
    )


def test_no_progress_guard_stops_renewal_so_the_wedged_lease_is_reclaimable(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    holder = f"pid={os.getpid()}:turn=wedge:platform=api_server"
    assert db.try_acquire_session_turn_lease("s1", holder, ttl_seconds=0.4) is True

    agent = _WedgeAgent()
    lease = DurableTurnLease(agent, db, "s1", holder)
    with lease._lock:
        lease.turn_active = True

    assert _watchdog(agent, lease, timeout_s=1.0)._tick() is False

    assert agent.interrupts, "the guard did not hard-interrupt the wedged turn"
    assert agent.interrupts[0][1] is True
    assert lease.stop.is_set() is True
    assert lease.is_turn_active() is False, "lease renewal was not stopped; the row would renew forever"

    next_holder = f"pid={os.getpid()}:turn=next:platform=api_server"
    deadline = time.monotonic() + 3.0
    acquired = False
    while time.monotonic() < deadline:
        if db.try_acquire_session_turn_lease("s1", next_holder, ttl_seconds=5):
            acquired = True
            break
        time.sleep(0.02)
    assert acquired, "the wedged lease never expired; the guard released nothing"
    db.release_session_turn_lease("s1", next_holder)


def test_guard_stays_quiet_while_the_turn_is_inside_the_threshold(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    holder = f"pid={os.getpid()}:turn=busy:platform=api_server"
    assert db.try_acquire_session_turn_lease("s1", holder, ttl_seconds=30) is True

    agent = _WedgeAgent(idle_seconds=5.0)
    lease = DurableTurnLease(agent, db, "s1", holder)
    with lease._lock:
        lease.turn_active = True

    assert _watchdog(agent, lease, timeout_s=900.0)._tick() is None
    assert agent.interrupts == []
    assert lease.is_turn_active() is True
    db.release_session_turn_lease("s1", holder)


def test_default_threshold_is_above_the_longest_measured_legitimate_park():
    """The guard must not kill a turn that is legitimately parked.

    The longest legitimate park measured on this host is a 455s compression
    pass, which makes no model call and completes no tool while it runs.
    """
    assert DEFAULT_TURN_LIVENESS_TIMEOUT_S > 455.0
