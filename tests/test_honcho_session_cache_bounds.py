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


# ---------------------------------------------------------------------------
# hard caps, unsynced buffers, peers, and read activity (follows #71463)
# ---------------------------------------------------------------------------

from plugins.memory.honcho.session import (  # noqa: E402
    _PEERS_CACHE_MAX_SIZE,
    _SESSION_CACHE_MAX_SIZE,
)


def _fill_sessions(mgr, count, unsynced_keys=()):
    for i in range(count):
        key = f"k{i}"
        session = _session(key=key)
        if key in unsynced_keys:
            session.add_message("user", "pending", _synced=False)
        mgr._cache[key] = session
        mgr._sessions_cache[session.honcho_session_id] = object()
        mgr._session_observation[session.honcho_session_id] = {"ai_observe_others": False}
        mgr._context_cache[key] = {"representation": "r"}


def test_size_cap_evicts_least_recently_used_sessions_with_their_entries():
    mgr = _manager()
    _fill_sessions(mgr, _SESSION_CACHE_MAX_SIZE + 2)

    with mgr._cache_lock:
        mgr._enforce_cache_caps_locked()

    assert len(mgr._cache) == _SESSION_CACHE_MAX_SIZE
    assert "k0" not in mgr._cache and "k1" not in mgr._cache
    assert "k2" in mgr._cache
    for gone in ("k0", "k1"):
        assert f"hs-{gone}" not in mgr._sessions_cache
        assert f"hs-{gone}" not in mgr._session_observation
        assert gone not in mgr._context_cache
    assert "hs-k2" in mgr._sessions_cache and "hs-k2" in mgr._session_observation


def test_size_cap_never_evicts_a_session_with_unsynced_messages():
    mgr = _manager()
    _fill_sessions(mgr, _SESSION_CACHE_MAX_SIZE + 1, unsynced_keys={"k0"})

    with mgr._cache_lock:
        mgr._enforce_cache_caps_locked()

    assert "k0" in mgr._cache  # the only copy until its flush lands
    assert "k1" not in mgr._cache
    assert len(mgr._cache) == _SESSION_CACHE_MAX_SIZE


def test_idle_sweep_keeps_sessions_with_unsynced_messages():
    mgr = _manager()
    stale = _session(key="stale")
    stale.add_message("user", "pending", _synced=False)
    stale.updated_at = datetime.now() - timedelta(seconds=_SESSION_IDLE_TTL_SECONDS + 1)
    mgr._cache = {"stale": stale}

    with mgr._cache_lock:
        evicted = mgr._sweep_idle_sessions_locked()

    assert evicted == 0
    assert "stale" in mgr._cache


def test_peers_cap_evicts_unreferenced_peers_oldest_first():
    mgr = _manager()
    live = _session(key="live")
    mgr._cache = {"live": live}
    mgr._peers_cache[live.user_peer_id] = object()
    mgr._peers_cache[live.assistant_peer_id] = object()
    for i in range(_PEERS_CACHE_MAX_SIZE):
        mgr._peers_cache[f"guest{i}"] = object()

    with mgr._cache_lock:
        mgr._enforce_cache_caps_locked()

    assert len(mgr._peers_cache) == _PEERS_CACHE_MAX_SIZE
    assert live.user_peer_id in mgr._peers_cache and live.assistant_peer_id in mgr._peers_cache
    assert "guest0" not in mgr._peers_cache and "guest1" not in mgr._peers_cache
    assert f"guest{_PEERS_CACHE_MAX_SIZE - 1}" in mgr._peers_cache


def test_recall_read_counts_as_activity_for_the_idle_sweep():
    mgr = _manager()
    session = _session(key="read-only")
    session.updated_at = datetime.now() - timedelta(seconds=_SESSION_IDLE_TTL_SECONDS + 1)
    mgr._cache = {"read-only": session}

    assert mgr._cached_session("read-only") is session
    with mgr._cache_lock:
        evicted = mgr._sweep_idle_sessions_locked()

    assert evicted == 0
    assert "read-only" in mgr._cache


def test_sdk_object_hit_moves_the_key_to_the_recent_end():
    mgr = _manager()
    mgr._peers_cache = {"a": object(), "b": object()}

    mgr._cached_sdk_object(mgr._peers_cache, "a", lambda: None)

    assert list(mgr._peers_cache) == ["b", "a"]


def test_get_or_create_stores_observation_flags_with_the_entry_and_eviction_drops_them():
    mgr = _manager()
    mgr._config.ai_peer = "hermes"
    mgr._config.peer_name = "operator"  # unnamed peers now fail closed instead of minting a fallback
    flags = {"user_observe_me": True, "user_observe_others": True, "ai_observe_me": True, "ai_observe_others": False}
    mgr._get_or_create_peer = lambda peer_id: object()
    mgr._get_or_create_honcho_session = lambda sid, user, assistant: (object(), [], dict(flags))

    session = mgr.get_or_create("cli:one")

    assert mgr._session_observation[session.honcho_session_id] == flags
    assert mgr._ai_observes_others(session) is False
    with mgr._cache_lock:
        mgr._evict_session_locked("cli:one", session)
    assert session.honcho_session_id not in mgr._session_observation


def test_flush_does_not_resurrect_an_evicted_session():
    mgr = _manager()
    session = _session(key="gone")
    session.add_message("user", "late", _synced=False)
    peer = SimpleNamespace(message=lambda content: content)
    mgr._get_or_create_peer = lambda peer_id: peer
    mgr._sessions_cache[session.honcho_session_id] = SimpleNamespace(add_messages=lambda messages: None)

    assert mgr._flush_session(session) is True
    assert "gone" not in mgr._cache
    assert all(m["_synced"] for m in session.messages)


def _sdk_session():
    return SimpleNamespace(add_messages=lambda messages: None)


def test_flush_that_recreates_the_sdk_session_stores_its_observation_flags():
    """After an eviction the flush path rebuilds the SDK session. The flags it configured must be kept."""
    mgr = _manager()
    session = _session(key="back")
    session.add_message("user", "hello", _synced=False)
    flags = {"user_observe_me": False, "user_observe_others": True, "ai_observe_me": True, "ai_observe_others": False}
    mgr._get_or_create_peer = lambda peer_id: SimpleNamespace(message=lambda content: content)
    mgr._get_or_create_honcho_session = lambda sid, user, assistant: (_sdk_session(), [], flags)

    assert mgr._flush_session(session) is True
    assert mgr._session_observation[session.honcho_session_id] == flags


def test_cached_sdk_session_returns_the_flags_stored_for_it():
    mgr = _manager()
    flags = {"user_observe_me": True, "user_observe_others": False, "ai_observe_me": True, "ai_observe_others": True}
    sdk = _sdk_session()
    mgr._sessions_cache["hs-x"] = sdk
    mgr._session_observation["hs-x"] = flags

    assert mgr._get_or_create_honcho_session("hs-x", None, None) == (sdk, [], flags)


def test_deferred_save_puts_an_evicted_session_back_in_the_cache():
    """write_frequency "session" defers to flush_all(), which only sees cached sessions."""
    mgr = _manager()
    mgr._write_frequency = "session"
    session = _session(key="evicted")
    session.add_message("user", "unsynced", _synced=False)

    mgr.save(session)

    assert mgr._cache["evicted"] is session


def test_deferred_save_of_a_fully_synced_evicted_session_stays_out_of_the_cache():
    mgr = _manager()
    mgr._write_frequency = "session"
    session = _session(key="done")
    session.add_message("user", "old", _synced=True)

    mgr.save(session)

    assert "done" not in mgr._cache


def test_deferred_save_flushes_inline_when_a_newer_object_owns_the_key():
    mgr = _manager()
    mgr._write_frequency = "session"
    newer = _session(key="k")
    mgr._cache["k"] = newer
    stale = _session(key="k")
    stale.add_message("user", "late", _synced=False)
    flushed = []
    mgr._flush_session = lambda s: flushed.append(s) or True

    mgr.save(stale)

    assert flushed == [stale]
    assert mgr._cache["k"] is newer
