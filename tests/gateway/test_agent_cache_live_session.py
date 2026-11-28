"""Regression for #154994: an agent's live session must agree with its cache snapshot."""

import threading
from collections import OrderedDict
from types import SimpleNamespace

from gateway.run_turn_runner import TurnRunner


KEY = "agent:main:local:dm:desk"


def test_live_session_drift_evicts_cached_agent_before_foreign_turn():
    chat_agent = SimpleNamespace(session_id="async-subagent", max_iterations=0)
    cache = OrderedDict({KEY: (chat_agent, "same-sig", None, "bot-chat")})
    init_calls = []
    runner = SimpleNamespace(
        _agent_cache=cache,
        _agent_cache_lock=threading.Lock(),
        _init_cached_agent_for_turn=lambda agent, depth: init_calls.append((agent, depth)),
    )
    turn = TurnRunner(
        runner, SimpleNamespace(session_key=KEY, session_id="bot-chat", _interrupt_depth=0)
    )

    found = turn._lookup_cached_agent(
        "same-sig", runner._agent_cache_lock, cache, 10, None, False, None
    )

    assert found.agent is None and not found.reused
    assert found.evicted is chat_agent
    assert KEY not in cache
    assert init_calls == []
