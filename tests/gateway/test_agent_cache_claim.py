"""The pending/startup window owns its cached agent before the async tracker catches up."""
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.codex_lifecycle import codex_retirements
from agent.transports.codex_app_server_session import CodexAppServerSession
from gateway.agent_cache_claim import owns_cached_agent
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.run_turn_runner import TurnRunner
from run_agent import AIAgent


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('raises', [False, True])
def test_checkout_protects_startup_until_worker_finally(monkeypatch, cached, raises):
    from gateway import agent_cache_pressure as pressure
    entered, proceed = threading.Event(), threading.Event()
    client = Mock()
    def request(*args, **kwargs):
        entered.set()
        assert proceed.wait(10)
        return {'thread': {'id': 'synthetic-thread'}}
    client.request = request
    session = CodexAppServerSession(resume_thread_id='synthetic-thread', client_factory=lambda **_: client)
    agent = AIAgent.__new__(AIAgent)
    agent._codex_session = session
    agent._last_activity_ts = 0
    agent.session_id = 'synthetic-session'
    agent.release_clients = Mock(side_effect=agent._close_codex_session)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = OrderedDict(lane=(agent, 'sig', 0, agent.session_id)) if cached else OrderedDict()
    runner._running_agents = {'lane': _AGENT_PENDING_SENTINEL}
    runner._init_cached_agent_for_turn = lambda *_: None
    runner._refresh_fallback_model = lambda: None
    runner._apply_fallback_chain_to_agent = lambda *_: None
    runner._extract_cache_busting_config = lambda _: {}
    runner._agent_config_signature = lambda *args, **kwargs: 'sig'
    runner._agent_cache_cap = lambda: 1
    runner._agent_cache_idle_ttl = lambda: 1
    runner._agent_cache_bounds = lambda: pressure.AgentCacheBounds(memory_high_mb=1, protect_recent=0)
    runner._spawn_release_thread = lambda fn, args, *_, **kwargs: fn(*args)
    monkeypatch.setattr(pressure, 'read_anon_rss_mb', lambda: 2)
    monkeypatch.setattr(pressure, 'transcript_persistence_caught_up', lambda _: True)
    ctx = SimpleNamespace(session_key='lane', session_id=agent.session_id, user_config={},
                          enabled_toolsets=[], source=SimpleNamespace(), _interrupt_depth=0)
    turn = TurnRunner(runner, ctx)
    turn._skip_context_files = lambda _: False
    turn._cached_sid_is_dead = lambda *_: (None, False)
    turn._current_message_count = lambda: 0
    turn._build_fresh_agent = lambda *_: agent
    @owns_cached_agent
    def execute(turn):
        got, _ = turn._resolve_turn_agent({'model': 'synthetic', 'runtime': {}}, 'feishu', '', 5, {}, {})
        assert got is agent
        session.ensure_started()
        if raises:
            raise ValueError('synthetic worker error')
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(execute, turn)
        try:
            assert entered.wait(5)
            assert runner._running_agent_ids() == set()  # tracker has not published the agent
            runner._evict_cached_agent('lane')
            assert runner._sweep_idle_cached_agents() == 0
            assert runner._sweep_agent_cache_under_pressure() == 0
            runner._agent_cache['other'] = (None, 'sig')
            with runner._agent_cache_lock:
                runner._enforce_agent_cache_cap()
            contender = TurnRunner(runner, ctx)
            with pytest.raises(RuntimeError, match='another turn'):
                contender._lookup_cached_agent('changed', runner._agent_cache_lock, runner._agent_cache, 5, None, False, 0)
            assert runner._agent_cache['lane'][0] is agent
            client.close.assert_not_called()
        finally:
            proceed.set()
        if raises:
            with pytest.raises(ValueError, match='synthetic worker error'):
                worker.result(timeout=5)
        else:
            worker.result(timeout=5)
    assert 'lane' not in runner._agent_cache
    client.close.assert_called_once()
    assert agent._codex_session is None


def test_resume_identity_survives_inflight_cache_retirement(tmp_path):
    entered, proceed = threading.Event(), threading.Event()
    class Client:
        closed = 0
        def initialize(self, **_):
            pass
        def request(self, method, params, **_):
            assert method == 'thread/resume'
            entered.set()
            assert proceed.wait(5)
            return {'thread': {'id': 'durable-thread'}}
        def close(self):
            self.closed += 1
            if self.closed == 1:
                raise TimeoutError('synthetic close failure')
    client = Client()
    session = CodexAppServerSession(codex_home=str(tmp_path), resume_thread_id='durable-thread',
                                   client_factory=lambda **_: client)
    original_key = session.writer_key
    agent = SimpleNamespace(_codex_session=session)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = OrderedDict(lane=(agent, 'sig'))
    runner._running_agents = {'lane': _AGENT_PENDING_SENTINEL}
    queued = []
    runner._spawn_release_thread = lambda *args, **kwargs: queued.append(args)
    with ThreadPoolExecutor(max_workers=1) as pool:
        startup = pool.submit(session.ensure_started)
        try:
            assert entered.wait(5)
            assert session.writer_key == original_key
            # Even another eviction caller without a checkout claim must capture the resume target.
            runner._evict_cached_agent('lane')
            assert queued and not runner._agent_cache
            proceed.set()
            startup.result(timeout=5)
            owners = codex_retirements(runner)
            with pytest.raises(TimeoutError, match='synthetic close failure'):
                owners.release(agent)
            owners.wait(original_key)
            assert agent._codex_session is None
            assert client.closed == 2
        finally:
            proceed.set()
            startup.result(timeout=5)
            session.close()
