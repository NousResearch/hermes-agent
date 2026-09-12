"""A rebuild that displaces a cached agent must release the one it replaced.

Every EVICTION path releases what it pops. A plain cache overwrite released
nothing, so the displaced agent's provider session was dropped to GC — and on
the claude-agent-sdk lane that means a live Claude Code CLI subprocess (~260 MB)
that GC can never reap. Measured before the fix: 13 turns produced 11 SDK
sessions but only 2 agent closes, leaving 11 orphans holding 2.9 GB.

The trigger is memory-pressure eviction, and BOTH the pressure sweep and
`_evict_cached_agent` are silent (no logger calls), so a regression here would
leave no trace in the logs. These tests are the only guard.
"""

from __future__ import annotations

import threading

import pytest


def _make_runner(running=()):
    """Minimal GatewayRunner with just the cache/teardown surface."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._released = []
    runner._running_agent_items = lambda: [("k", a) for a in running]
    # Capture instead of tearing down: the real one closes sockets/subprocesses.
    runner._release_evicted_agent_soft = runner._released.append
    return runner


class _Agent:
    def __init__(self, name="a"):
        self.name = name


def _settle():
    """The release runs on a daemon thread; give it a moment to land."""
    for t in threading.enumerate():
        if t.name.startswith("agent-displaced-"):
            t.join(timeout=5)


def test_displaced_agent_is_released():
    runner = _make_runner()
    old = _Agent("old")

    runner._release_displaced_agent(old, "sess-1")
    _settle()

    assert runner._released == [old]


def test_displaced_agent_release_uses_owning_profile_scope(tmp_path, monkeypatch):
    """A secondary profile's SDK teardown must not inherit the launch profile's home or secrets."""
    from types import SimpleNamespace

    from agent import secret_scope
    from hermes_constants import get_hermes_home

    default_home = tmp_path / ".hermes"
    profile_home = default_home / "profiles" / "b"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text("HINDSIGHT_LLM_API_KEY=key-of-b\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    runner = _make_runner()
    session_key = "agent:b:telegram:dm:1"
    runner.session_store = SimpleNamespace(_profile_home_for_key=lambda key: profile_home)
    seen = {}
    done = threading.Event()

    def _release(agent):
        seen["agent"] = agent
        seen["scope"] = secret_scope.current_secret_scope()
        seen["home"] = get_hermes_home()
        done.set()

    runner._release_evicted_agent_soft = _release
    old = _Agent("old")
    secret_scope.set_multiplex_active(True)
    try:
        runner._release_displaced_agent(old, session_key)
        assert done.wait(5)
    finally:
        secret_scope.set_multiplex_active(False)

    assert seen["agent"] is old
    assert seen["home"] == profile_home
    assert seen["scope"] and seen["scope"].get("HINDSIGHT_LLM_API_KEY") == "key-of-b"


def test_mid_turn_agent_is_never_torn_down():
    """Its own completion path owns teardown — releasing it would kill a live turn."""
    busy = _Agent("busy")
    runner = _make_runner(running=(busy,))

    runner._release_displaced_agent(busy, "sess-1")
    _settle()

    assert runner._released == []


def test_none_and_sentinel_are_ignored():
    from gateway.run import _AGENT_PENDING_SENTINEL

    runner = _make_runner()

    runner._release_displaced_agent(None, "sess-1")
    runner._release_displaced_agent(_AGENT_PENDING_SENTINEL, "sess-1")
    _settle()

    assert runner._released == []


def test_release_still_happens_when_running_lookup_fails():
    """A broken running-set lookup must not silently strand the agent."""
    runner = _make_runner()
    old = _Agent("old")

    def _boom():
        raise RuntimeError("registry unavailable")

    runner._running_agent_items = _boom

    runner._release_displaced_agent(old, "sess-1")
    _settle()

    assert runner._released == [old]


def test_falls_back_to_inline_release_when_thread_spawn_fails(monkeypatch):
    """At interpreter shutdown a thread cannot start; the agent must still go."""
    runner = _make_runner()
    old = _Agent("old")

    def _no_threads(*args, **kwargs):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading, "Thread", _no_threads)

    runner._release_displaced_agent(old, "sess-1")

    assert runner._released == [old]


def test_a_raising_release_is_contained():
    """Teardown failure must not propagate into the turn that displaced it."""
    runner = _make_runner()

    def _boom(agent):
        raise RuntimeError("socket teardown failed")

    runner._release_evicted_agent_soft = _boom

    runner._release_displaced_agent(_Agent("old"), "sess-1")
    _settle()  # must not raise
