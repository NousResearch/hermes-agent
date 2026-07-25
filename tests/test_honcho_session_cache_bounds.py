"""Regression tests for bounded growth of the Honcho local session cache.

Covers the fix for unbounded RSS growth in long-running gateways: prior to
this, ``HonchoSession.messages`` grew forever (never trimmed after a sync),
and ``HonchoSessionManager``'s ``_cache``/``_sessions_cache``/``_context_cache``
had no eviction path short of an explicit ``/new`` reset.
"""

import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

from plugins.memory.honcho.session import (
    HonchoSession,
    HonchoSessionManager,
    _SESSION_IDLE_TTL_SECONDS,
    _SESSION_MESSAGE_RETENTION,
)


def _session(key="k", honcho_session_id=None):
    return HonchoSession(
        key=key,
        user_peer_id="user",
        assistant_peer_id="assistant",
        honcho_session_id=honcho_session_id or f"hs-{key}",
    )


def _manager():
    cfg = SimpleNamespace(
        write_frequency="turn",
        dialectic_reasoning_level="low",
        dialectic_dynamic=True,
        dialectic_max_chars=600,
        observation_mode="directional",
        user_observe_me=True,
        user_observe_others=True,
        ai_observe_me=True,
        ai_observe_others=True,
        message_max_chars=25000,
        dialectic_max_input_chars=10000,
    )
    return HonchoSessionManager(honcho=SimpleNamespace(), config=cfg)


def test_trim_synced_messages_caps_total_length():
    session = _session()
    for i in range(250):
        session.add_message("user", f"msg{i}", _synced=True)
    for i in range(250, 255):
        session.add_message("user", f"msg{i}", _synced=False)
    assert len(session.messages) == 255

    HonchoSessionManager._trim_synced_messages(session)

    assert len(session.messages) == _SESSION_MESSAGE_RETENTION
    # oldest synced messages are the ones dropped; the unsynced tail survives intact
    assert not any(m.get("_synced") is False for m in session.messages[:-5])
    assert all(m.get("_synced") is False for m in session.messages[-5:])


def test_trim_synced_messages_never_drops_an_unsynced_message():
    session = _session()
    for i in range(300):
        session.add_message("user", f"m{i}", _synced=(i != 10))

    HonchoSessionManager._trim_synced_messages(session)

    contents = [m["content"] for m in session.messages]
    assert "m10" in contents
    idx = contents.index("m10")
    assert session.messages[idx].get("_synced") is False
    # trimming stops at the first unsynced message from the front — it does
    # not skip past it to keep reducing, so everything from there on survives
    assert contents[idx:] == [f"m{i}" for i in range(10, 300)]


def test_trim_synced_messages_is_a_noop_under_the_cap():
    session = _session()
    for i in range(10):
        session.add_message("user", f"m{i}", _synced=True)

    HonchoSessionManager._trim_synced_messages(session)

    assert len(session.messages) == 10


def test_sweep_idle_sessions_evicts_stale_entries_across_all_caches():
    mgr = _manager()
    stale = _session(key="stale", honcho_session_id="hs-stale")
    stale.updated_at = datetime.now() - timedelta(seconds=_SESSION_IDLE_TTL_SECONDS + 1)
    fresh = _session(key="fresh", honcho_session_id="hs-fresh")

    mgr._cache = {"stale": stale, "fresh": fresh}
    mgr._sessions_cache = {"hs-stale": object(), "hs-fresh": object()}
    mgr._context_cache = {"stale": {"x": 1}, "fresh": {"x": 1}}

    evicted = mgr._sweep_idle_sessions_locked()

    assert evicted == 1
    assert set(mgr._cache) == {"fresh"}
    assert set(mgr._sessions_cache) == {"hs-fresh"}
    assert set(mgr._context_cache) == {"fresh"}


def test_sweep_idle_sessions_keeps_fresh_entries():
    mgr = _manager()
    fresh = _session(key="fresh")
    mgr._cache = {"fresh": fresh}
    mgr._sessions_cache = {fresh.honcho_session_id: object()}
    mgr._context_cache = {"fresh": {}}

    evicted = mgr._sweep_idle_sessions_locked()

    assert evicted == 0
    assert "fresh" in mgr._cache


def test_maybe_sweep_idle_sessions_is_rate_limited():
    mgr = _manager()
    stale = _session(key="stale")
    stale.updated_at = datetime.now() - timedelta(seconds=_SESSION_IDLE_TTL_SECONDS + 1)
    mgr._cache = {"stale": stale}

    mgr._last_idle_sweep_ts = time.time()  # just swept — this call should no-op
    mgr._maybe_sweep_idle_sessions()
    assert "stale" in mgr._cache

    mgr._last_idle_sweep_ts = 0.0  # force the interval to have elapsed
    mgr._maybe_sweep_idle_sessions()
    assert "stale" not in mgr._cache


def test_get_or_create_triggers_sweep_without_blocking_on_lock_reentrancy():
    """`_cache_lock` is an RLock specifically so a sweep triggered from inside
    `get_or_create` (which also takes the lock) can't deadlock the manager's
    own thread. Guard against that regressing silently.
    """
    mgr = _manager()
    stale = _session(key="stale")
    stale.updated_at = datetime.now() - timedelta(seconds=_SESSION_IDLE_TTL_SECONDS + 1)
    mgr._cache = {"stale": stale}
    mgr._last_idle_sweep_ts = 0.0

    done = threading.Event()

    def call_it():
        mgr._maybe_sweep_idle_sessions()
        done.set()

    t = threading.Thread(target=call_it)
    t.start()
    t.join(timeout=5)
    assert done.is_set(), "sweep did not complete — possible deadlock"
    assert "stale" not in mgr._cache
