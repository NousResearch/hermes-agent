"""Regression: _session_turn_lease_active must unwrap the AsyncSessionDB door (2026-09-15 watcher loop).
The gateway sets runner._session_db = AsyncSessionDB(store); the probe runs inside to_thread and
must call the raw store — via the door it got a coroutine and raised on tuple-unpack every 2 s tick."""
import time
from hermes_state import AsyncSessionDB
from gateway.run_notifications import GatewayNotificationsMixin

class _Store:
    def __init__(self, owner): self._owner = owner
    def get_session_turn_lease_owner(self, session_id): return self._owner

class _Runner(GatewayNotificationsMixin):
    def __init__(self, db): self._session_db = db

def test_async_door_live_lease_is_true():
    r = _Runner(AsyncSessionDB(_Store(("h", time.time() + 60))))
    assert r._session_turn_lease_active("s1") is True

def test_async_door_expired_lease_is_false():
    r = _Runner(AsyncSessionDB(_Store(("h", time.time() - 60))))
    assert r._session_turn_lease_active("s1") is False

def test_raw_store_still_works():
    r = _Runner(_Store(("h", time.time() + 60)))
    assert r._session_turn_lease_active("s1") is True

def test_none_owner_and_bad_shape_fail_open():
    assert _Runner(AsyncSessionDB(_Store(None)))._session_turn_lease_active("s1") is False
    assert _Runner(AsyncSessionDB(_Store("garbage")))._session_turn_lease_active("s1") is False
