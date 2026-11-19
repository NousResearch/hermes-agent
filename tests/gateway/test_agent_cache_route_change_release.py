"""A turn on another route (one-turn ``/moa``, a refreshed credential) replaces the cached agent, so
the agent it displaces is handed back for release instead of being orphaned with live clients (a
Codex app-server child and its reader threads)."""

import threading
from collections import OrderedDict
from types import SimpleNamespace

from gateway.run_turn_runner import TurnRunner

KEY = "agent:main:local:dm:desk"


def _lookup(cache, sig):
    runner = SimpleNamespace(_agent_cache=cache, _agent_cache_lock=threading.Lock(),
                             _init_cached_agent_for_turn=lambda agent, depth: None)
    turn = TurnRunner(runner, SimpleNamespace(session_key=KEY, session_id="sid-1", _interrupt_depth=0))
    return turn._lookup_cached_agent(sig, runner._agent_cache_lock, cache, 10, None, False, None)


def test_route_change_hands_the_displaced_agent_back_for_release():
    chat_agent = SimpleNamespace(name="chat")
    cache = OrderedDict({KEY: (chat_agent, "sig-chat", None, "sid-1")})

    found = _lookup(cache, "sig-moa")

    assert found.agent is None and not found.reused
    assert found.evicted is chat_agent
    assert KEY not in cache


def test_same_route_reuses_and_evicts_nothing():
    chat_agent = SimpleNamespace(name="chat", max_iterations=0)
    cache = OrderedDict({KEY: (chat_agent, "sig-chat", None, "sid-1")})

    found = _lookup(cache, "sig-chat")

    assert found.agent is chat_agent and found.reused and found.evicted is None
