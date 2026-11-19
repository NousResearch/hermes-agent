"""A finished turn's lease and timers free by reference counting, not only by a full GC.

The gateway's timer trims no longer force a full ``gc.collect()`` (it froze every session for
250-750 ms). Python 3.14's incremental collector advances only on new allocations, so on a quiet
gateway anything held by a reference cycle stays resident: each turn's lease <-> watchdog and
owner -> scheduler handle -> bound method -> owner cycles used to pin the whole agent (history,
client, compressor) after it closed.
"""

import gc
import threading
import weakref

from agent.periodic_scheduler import schedule
from agent.turn_facade_lease import DurableTurnLease


class _Db:
    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        return True

    def release_session_turn_lease(self, session_id, holder):
        pass


class _Agent:
    def __init__(self):
        self.session_id = "s1"
        self._session_turn_lease_refresh_interval = 60.0
        self._active_session_turn_lease_holder = None
        self._lock = threading.Lock()

    def _touch_activity(self, *args, **kwargs):
        pass

    def _liveness_activity_lock(self):
        return self._lock


def test_finished_turn_lease_does_not_keep_the_agent_alive_without_gc():
    gc.disable()
    try:
        agent = _Agent()
        lease = DurableTurnLease(agent, _Db(), "s1", "h")
        lease.build_threads()
        assert lease.watchdog is not None  # the liveness watchdog is on by default
        lease.start()
        lease.stop_refresher()
        lease.join_threads()
        lease.clear_interrupt()
        lease.release()
        agent_ref = weakref.ref(agent)
        del lease, agent
        assert agent_ref() is None
    finally:
        gc.enable()


def test_cancelled_handle_releases_its_owner_without_gc():
    class Owner:
        handle = None

        def tick(self):
            return None

    gc.disable()
    try:
        owner = Owner()
        owner.handle = schedule(owner.tick, 3600)
        owner.handle.cancel()
        owner_ref = weakref.ref(owner)
        del owner
        assert owner_ref() is None
    finally:
        gc.enable()
