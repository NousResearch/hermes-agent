"""Independent deterministic schedules through the real cache claim helpers."""
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.agent_cache_claim import claim_cached_agent, release_cached_agent
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.run_turn_runner import TurnRunner


def test_sync_finally_survives_async_running_slot_cleanup():
    released, proceed = threading.Event(), threading.Event()
    class ExitGate:
        def __init__(self):
            self.lock = threading.Lock()
            self.worker = None
        def __enter__(self):
            self.lock.acquire()
            return self
        def __exit__(self, *_):
            self.lock.release()
            if threading.get_ident() == self.worker:
                self.worker = None
                released.set()
                assert proceed.wait(5)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache_lock = ExitGate()
    agent = SimpleNamespace(_codex_session=None, _cache_evict_after_turn=True)
    runner._agent_cache = OrderedDict(lane=(agent, 'sig'))
    runner._running_agents = {'lane': agent}
    runner._persist_active_agents = lambda: None
    runner._spawn_release_thread = lambda *a, **kw: None
    turn = SimpleNamespace(_runner=runner, _ctx=SimpleNamespace(session_key='lane'))
    with runner._agent_cache_lock:
        claim_cached_agent(turn, agent)
    def finish_worker():
        runner._agent_cache_lock.worker = threading.get_ident()
        release_cached_agent(turn)
    with ThreadPoolExecutor(max_workers=1) as pool:
        finished = pool.submit(finish_worker)
        try:
            assert released.wait(5)
            runner._release_running_agent_state('lane')
            assert '_cache_evict_after_turn' not in vars(agent)
        finally:
            proceed.set()
        finished.result(timeout=5)


def test_two_cold_misses_cannot_publish_two_claimed_owners():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = OrderedDict()
    runner._running_agents = {'lane': _AGENT_PENDING_SENTINEL}
    runner._persist_active_agents = lambda: None
    runner._extract_cache_busting_config = lambda _: {}
    runner._agent_config_signature = lambda *args, **kwargs: 'sig'
    runner._enforce_agent_cache_cap = lambda: None
    old_building, finish_old_build = threading.Event(), threading.Event()
    turns, agents = [], []
    for index in range(2):
        agent = SimpleNamespace(_codex_session=None)
        ctx = SimpleNamespace(session_key='lane', session_id='same-session', user_config={},
                              enabled_toolsets=[], source=SimpleNamespace(), _interrupt_depth=0,
                              run_generation=index + 1)
        turn = TurnRunner(runner, ctx)
        turn._skip_context_files = lambda _: False
        turn._cached_sid_is_dead = lambda *_: (None, False)
        turn._current_message_count = lambda: 0
        def build(*_, agent=agent, index=index):
            if index == 0:
                old_building.set()
                assert finish_old_build.wait(5)
            return agent
        turn._build_fresh_agent = build
        turns.append(turn)
        agents.append(agent)
    successes, rejected = [], []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            old = pool.submit(turns[0]._resolve_turn_agent,
                              {'model': 'synthetic', 'runtime': {}}, 'feishu', '', 5, {}, {})
            assert old_building.wait(5)
            # /stop's real pending-slot release permits a successor while the old
            # synchronous constructor still has no published interruptible agent.
            runner._drop_turn_slot('lane')
            new = pool.submit(turns[1]._resolve_turn_agent,
                              {'model': 'synthetic', 'runtime': {}}, 'feishu', '', 5, {}, {})
            assert new.result(timeout=5)[0] is agents[1]
            assert runner._agent_cache['lane'][0] is agents[1]
            finish_old_build.set()
            futures = [old, new]
            for future in futures:
                try:
                    successes.append(future.result(timeout=5))
                except RuntimeError as exc:
                    rejected.append(str(exc))
        assert len(successes) == 1
        assert len(rejected) == 1
        assert runner._agent_cache['lane'][0] is agents[1]
        assert sum('_gateway_cache_owner' in vars(a) for a in agents) == 1
    finally:
        finish_old_build.set()
        for turn in turns:
            release_cached_agent(turn)


def test_pending_slot_release_accepts_agent_sentinel():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = OrderedDict()
    runner._running_agents = {'lane': _AGENT_PENDING_SENTINEL}
    runner._persist_active_agents = lambda: None
    runner._drop_turn_slot('lane')
    assert runner._peek_session_state('lane').turn.agent is None


def test_invalidated_construction_releases_unpublished_owner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = OrderedDict()
    runner._session_run_generation = {'lane': 1}
    runner._extract_cache_busting_config = lambda _: {}
    runner._agent_config_signature = lambda *args, **kwargs: 'sig'
    session = Mock(writer_key=('synthetic-home', 'synthetic-thread'))
    agent = SimpleNamespace(_codex_session=session, release_clients=Mock())
    ctx = SimpleNamespace(session_key='lane', session_id='synthetic-session', user_config={},
                          enabled_toolsets=[], source=SimpleNamespace(), _interrupt_depth=0,
                          run_generation=1)
    turn = TurnRunner(runner, ctx)
    turn._skip_context_files = lambda _: False
    turn._cached_sid_is_dead = lambda *_: (None, False)
    turn._current_message_count = lambda: 0
    def build(*_):
        runner._begin_session_run_generation('lane')
        return agent
    turn._build_fresh_agent = build
    with pytest.raises(RuntimeError, match='superseded'):
        turn._resolve_turn_agent({'model': 'synthetic', 'runtime': {}}, 'feishu', '', 5, {}, {})
    session.close.assert_called_once()
    agent.release_clients.assert_called_once()
    assert agent._codex_session is None
    assert not runner._agent_cache
    assert not runner._codex_retirements._owners
    assert '_gateway_cache_owner' not in vars(agent)


def test_late_eviction_does_not_touch_successor_or_its_prompt_state():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    old, successor = SimpleNamespace(), SimpleNamespace()
    runner._agent_cache = OrderedDict(lane=(successor, 'sig'))
    state = runner._session_state('lane')
    state.conversation.ephemeral_pin = 'synthetic-pin'
    state.conversation.vc_last = 'synthetic-voice'
    runner._evict_cached_agent('lane', expected_agent=old)
    assert runner._agent_cache['lane'][0] is successor
    assert state.conversation.ephemeral_pin == 'synthetic-pin'
    assert state.conversation.vc_last == 'synthetic-voice'
