"""One-turn routes disappear immediately without retiring a still-running physical writer."""
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.codex_lifecycle import codex_retirements
from gateway.agent_cache_claim import claim_cached_agent, owns_cached_agent
from gateway.run import GatewayRunner


@pytest.mark.parametrize('commit_memory', [False, True])
def test_detached_one_turn_writer_survives_early_async_slot_release(commit_memory):
    entered, finish = threading.Event(), threading.Event()
    session = Mock(writer_key=('synthetic-home', 'synthetic-thread'))
    agent = SimpleNamespace(_codex_session=session, release_clients=Mock())
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = OrderedDict(lane=(agent, 'one-turn'))
    runner._running_agents = {'lane': agent}
    runner._persist_active_agents = lambda: None
    runner._spawn_release_thread = lambda target, args, *a, **kw: target(*args)
    runner._commit_memory_before_soft_evict = Mock()
    turn = SimpleNamespace(_runner=runner, _ctx=SimpleNamespace(session_key='lane'))

    @owns_cached_agent
    def worker(turn):
        with runner._agent_cache_lock:
            claim_cached_agent(turn, agent)
        entered.set()
        assert finish.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker, turn)
        try:
            assert entered.wait(5)
            runner._evict_cached_agent('lane', release_after_turn=True, commit_memory=commit_memory)
            assert 'lane' not in runner._agent_cache  # no stale one-turn route in Desktop/resume
            runner._release_running_agent_state('lane')  # cancelled asyncio wrapper unwinds early
            session.close.assert_not_called()
            agent.release_clients.assert_not_called()
            owners = codex_retirements(runner)
            with pytest.raises(RuntimeError, match='still executing'):
                owners.wait(session.writer_key)
            assert owners._owners[id(agent)][1] is session
            session.close.assert_not_called()
        finally:
            finish.set()
        future.result(timeout=5)
    session.close.assert_called_once()
    agent.release_clients.assert_called_once()
    assert agent._codex_session is None
    assert not runner._peek_session_state('lane').turn.retired_agents
    assert not owners._owners
    owners.wait(session.writer_key)  # same-thread resume is admitted after physical owner exit
    if commit_memory:
        runner._commit_memory_before_soft_evict.assert_called_once_with(agent, 'lane', True)
    else:
        runner._commit_memory_before_soft_evict.assert_not_called()
