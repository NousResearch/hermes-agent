"""Synthetic writer-lock regression through the real cache and API worker paths.

No credentials, real conversations or model requests: the stdio subprocess owns
an exclusive SQLite transaction until it exits, just as a Codex thread writer does.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import suppress
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.transports.codex_app_server_session import CodexAppServerSession
from agent.codex_lifecycle import codex_retirements, wait_for_codex_writer
from agent.codex_runtime import _start_codex_thread
from gateway.agent_cache_claim import owns_cached_agent
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from run_agent import AIAgent


@pytest.fixture
def writer_factory(tmp_path):
    script = tmp_path / 'synthetic-codex'
    script.write_text(f'#!{sys.executable}\n' + '''import json, os, sqlite3, sys
conn = None
for line in sys.stdin:
    msg = json.loads(line)
    if 'id' not in msg: continue
    method = msg['method']
    if method in ('thread/start', 'thread/resume'):
        try:
            conn = sqlite3.connect(os.path.join(os.environ['CODEX_HOME'], 'writer.sqlite'), timeout=0)
            conn.execute('BEGIN EXCLUSIVE')
            result = {'thread': {'id': 'synthetic-thread'}}
        except sqlite3.OperationalError:
            print(json.dumps({'id': msg['id'], 'error': {'code': -32600, 'message': 'already has an active writer'}}), flush=True)
            continue
    else: result = {}
    print(json.dumps({'id': msg['id'], 'result': result}), flush=True)
''')
    script.chmod(0o700)
    sessions, processes = [], []

    def create(*, delayed=False, start=True):
        entered, proceed = threading.Event(), threading.Event()

        class Session(CodexAppServerSession):
            def ensure_started(self):
                try:
                    return super().ensure_started()
                finally:
                    if self._client and self._client._proc not in processes:
                        processes.append(self._client._proc)

            def close(self):
                if delayed and not self._closed:
                    entered.set()
                    assert proceed.wait(10), 'test never released the retirement gate'
                super().close()

        session = Session(cwd=str(tmp_path), codex_bin=str(script), codex_home=str(tmp_path),
                          resume_thread_id='synthetic-thread', model='synthetic-model')
        session.entered, session.proceed = entered, proceed
        sessions.append(session)
        if start:
            assert session.ensure_started() == 'synthetic-thread'
        return session

    yield create, processes
    for session in sessions:
        session.proceed.set()
        with suppress(Exception):
            session.close()
    for proc in processes:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


def agent_shell(session):
    agent = AIAgent.__new__(AIAgent)
    agent._codex_session = session
    agent.session_id = 'synthetic-session'
    agent._session_messages = []
    agent._last_activity_ts = 0
    # Only the Codex owner may be closed; broad lifecycle hooks are observable.
    agent.release_clients = Mock(side_effect=agent._close_codex_session)
    agent.close = Mock()
    agent.shutdown_memory_provider = Mock()
    return agent


@pytest.mark.platforms("posix")
@pytest.mark.parametrize('channel', ['feishu', 'slack'])
@pytest.mark.parametrize('reason', ['signature', 'messages', 'explicit', 'ttl', 'lru', 'pressure'])
def test_gateway_reuses_warm_writer_and_fences_replacement(tmp_path, monkeypatch, writer_factory, channel, reason):
    create, processes = writer_factory
    old = agent_shell(create(delayed=True))
    session = old._codex_session
    runner = GatewayRunner.__new__(GatewayRunner)
    key = f'agent:main:{channel}:synthetic'
    runner._agent_cache = OrderedDict({key: (old, 'same', 0, old.session_id)})
    runner._agent_cache_lock = threading.Lock()
    runner._session_db = None
    runner._peek_session_state = lambda _: None
    runner._running_agent_ids = lambda: set()
    runner._init_cached_agent_for_turn = lambda *_: None
    runner._refresh_fallback_model = lambda: None
    runner._apply_fallback_chain_to_agent = lambda *_: None
    runner._extract_cache_busting_config = lambda _: {}
    runner._agent_config_signature = lambda *_a, **_kw: 'same'
    runner._commit_memory_before_soft_evict = lambda *_: None
    runner._agent_cache_cap = lambda: 10
    ctx = SimpleNamespace(session_key=key, session_id=old.session_id, user_config={},
                          enabled_toolsets=[], source=SimpleNamespace(), _interrupt_depth=0)
    turn = TurnRunner(runner, ctx)
    turn._skip_context_files = lambda _: False
    built = []

    def build(*_):
        replacement = agent_shell(create(start=False))
        built.append(replacement)
        return replacement

    turn._build_fresh_agent = build
    route = {'model': 'synthetic-model', 'runtime': {}}
    @owns_cached_agent
    def resolve_turn(turn):
        result = turn._resolve_turn_agent(route, channel, '', 5, {}, {})
        wait_for_codex_writer(result[0])
        assert _start_codex_thread(result[0]) == 'synthetic-thread'
        return result

    resolve = lambda: resolve_turn(turn)
    for _ in range(3):
        assert resolve() == (old, True)
        assert old._codex_session.ensure_started() == 'synthetic-thread'
    assert len(processes) == 1
    old.release_clients.assert_not_called()

    def idle():
        runner._agent_cache_idle_ttl = lambda: 1
        assert runner._sweep_idle_cached_agents() == 1

    def lru():
        runner._agent_cache_cap = lambda: 1
        runner._agent_cache['unrelated'] = (None, 'same')
        with runner._agent_cache_lock:
            runner._enforce_agent_cache_cap()

    def pressure_evict():
        from gateway import agent_cache_pressure as pressure
        runner._agent_cache_bounds = lambda: pressure.AgentCacheBounds(memory_high_mb=1, protect_recent=0)
        monkeypatch.setattr(pressure, 'read_anon_rss_mb', lambda: 2)
        monkeypatch.setattr(pressure, 'transcript_persistence_caught_up', lambda _: True)
        assert runner._sweep_agent_cache_under_pressure() == 1

    actions = {
        'signature': lambda: setattr(runner, '_agent_config_signature', lambda *_a, **_kw: 'changed'),
        'messages': lambda: setattr(turn, '_current_message_count', lambda: 1),
        'explicit': lambda: runner._evict_cached_agent(key),
        'ttl': idle, 'lru': lru, 'pressure': pressure_evict,
    }
    actions[reason]()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(resolve)
        try:
            assert session.entered.wait(5), 'replacement bypassed retirement'
            with pytest.raises(TimeoutError):
                future.result(timeout=2)
            assert len(processes) == 1
            # Retirement waits without blocking unrelated cache operations.
            assert runner._agent_cache_lock.acquire(timeout=2)
            runner._agent_cache_lock.release()
        finally:
            session.proceed.set()
        new, reused = future.result(timeout=5)
    assert not reused and new is not old
    assert processes[0].poll() is not None
    assert sum(p.poll() is None for p in processes) == 1
    old._close_codex_session()  # repeated cleanup must not touch the replacement
    assert new._codex_session.ensure_started() == 'synthetic-thread'
    new._close_codex_session()
    # A subsequent owner can resume that same thread, without accumulating children.
    for _ in range(3):
        again = create()
        again.close()
        assert all(p.poll() is not None for p in processes)
    old.close.assert_not_called()
    old.shutdown_memory_provider.assert_not_called()


def test_unconfirmed_process_exit_keeps_replacement_fenced(monkeypatch):
    """A kill/wait failure must reach the successor, not look like a released writer."""
    import subprocess
    from agent.transports import codex_app_server as transport

    client = transport.CodexAppServerClient.__new__(transport.CodexAppServerClient)
    client._closed = False
    client._close_lock = threading.Lock()
    client._close_complete = False
    client._closing_descendants = None
    client._pending = {}
    client._pending_lock = threading.Lock()
    client._proc = Mock(stdin=None, pid=123456, poll=Mock(return_value=None),
                        wait=Mock(side_effect=subprocess.TimeoutExpired('synthetic-codex', 1)))
    monkeypatch.setattr(transport, '_snapshot_descendants', lambda _: [])
    session = CodexAppServerSession()
    session._client = client
    session._thread_id = "synthetic-thread"
    agent = agent_shell(session)
    runner = GatewayRunner.__new__(GatewayRunner)
    barrier = codex_retirements(runner)
    barrier.track(agent)
    with pytest.raises(subprocess.TimeoutExpired):
        barrier.release(agent)
    with pytest.raises(subprocess.TimeoutExpired):
        barrier.wait(session.writer_key)
    agent.release_clients.assert_not_called()
    # No poisoned routing-key Future: a later attempt confirms this same Popen exited.
    assert agent._codex_session is session
    client._proc.poll.return_value = 0
    client._proc.wait.side_effect = None
    barrier.wait(session.writer_key)
    assert agent._codex_session is None
    signals = client._proc.terminate.call_count + client._proc.kill.call_count
    barrier.release(agent)
    session.close()
    assert client._proc.terminate.call_count + client._proc.kill.call_count == signals


@pytest.mark.platforms("posix")
def test_concurrent_close_joins_the_original_writer(monkeypatch, writer_factory):
    create, processes = writer_factory
    session = create()
    agent = agent_shell(session)
    owner = codex_retirements(GatewayRunner.__new__(GatewayRunner))
    owner.track(agent)
    key = session.writer_key
    proc = processes[0]
    terminate, wait, stdin = proc.terminate, proc.wait, proc.stdin
    entered, proceed = threading.Event(), threading.Event()

    def delayed_wait(timeout=None):
        entered.set()
        assert proceed.wait(10)
        terminate()
        return wait(timeout=timeout)

    monkeypatch.setattr(proc, 'stdin', None)
    monkeypatch.setattr(proc, 'terminate', lambda: None)
    monkeypatch.setattr(proc, 'wait', delayed_wait)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(agent._close_codex_session)
        assert entered.wait(5)
        second = pool.submit(owner.wait, key)
        try:
            with pytest.raises(TimeoutError):
                second.result(timeout=2)
            assert agent._codex_session is session
            assert proc.poll() is None
            # A different Codex thread/home is independent of this closing owner.
            owner.wait((key[0], 'other-thread'))
            owner.wait((key[0] + '-other', key[1]))
        finally:
            proceed.set()
        first.result(timeout=5)
        second.result(timeout=5)
    stdin.close()
    assert proc.poll() is not None
    assert agent._codex_session is None
    resumed = create()
    resumed.close()


def test_pressure_retires_all_writers_before_slow_memory(monkeypatch):
    from gateway import agent_cache_pressure as pressure
    entered, proceed, done = threading.Event(), threading.Event(), threading.Event()
    a, b = (agent_shell(CodexAppServerSession()) for _ in range(2))
    a._codex_session._thread_id = 'one'
    b._codex_session._thread_id = 'two'
    a._codex_session._client = Mock()
    b._codex_session._client = Mock()
    first_client, second_client = a._codex_session._client, b._codex_session._client
    def slow_memory(_messages):
        entered.set()
        assert proceed.wait(10)
    a._memory_manager = SimpleNamespace(on_session_end=slow_memory)
    b._memory_manager = None
    a.context_compressor = b.context_compressor = None
    b.release_clients = Mock(side_effect=done.set)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = OrderedDict(first=(a, 'sig'), second=(b, 'sig'))
    runner._running_agent_ids = lambda: set()
    runner._agent_cache_bounds = lambda: pressure.AgentCacheBounds(memory_high_mb=1, protect_recent=0)
    runner._run_release_in_profile_scope = lambda fn, args, _key: fn(*args)
    monkeypatch.setattr(pressure, 'read_anon_rss_mb', lambda: 2)
    monkeypatch.setattr(pressure, 'transcript_persistence_caught_up', lambda _: True)
    assert runner._sweep_agent_cache_under_pressure() == 2
    try:
        assert entered.wait(5)
        first_client.close.assert_called_once()
        second_client.close.assert_called_once()
        assert b._codex_session is None
    finally:
        proceed.set()
        assert done.wait(5)


def test_soft_retirement_leaves_registered_review_to_finish():
    from agent.background_review import _track_review_fork
    parent = AIAgent.__new__(AIAgent)
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._background_review_agent = None
    parent._background_review_lock = threading.Lock()
    parent._codex_session = CodexAppServerSession()
    parent._drop_shared_client = Mock()
    parent._close_request_clients = Mock()
    review = Mock()
    _track_review_fork(parent, review, register=True)
    parent.release_clients()
    review.release_clients.assert_not_called()
    review.close.assert_not_called()
    assert review in parent._active_children
    assert parent._background_review_agent is review
    # The normal review finalizer still owns cleanup and unregisters by identity.
    _track_review_fork(parent, review, register=False)
    review.release_clients()
    assert not parent._active_children
    assert parent._background_review_agent is None


def test_running_eviction_keeps_owner_until_turn_release():
    session = CodexAppServerSession()
    session._thread_id = 'synthetic-thread'
    session._client = Mock()
    client = session._client
    agent = agent_shell(session)
    runner = GatewayRunner.__new__(GatewayRunner)
    turn = SimpleNamespace(agent=agent, lease=None)
    turn.clear = lambda: setattr(turn, 'agent', None)
    state = SimpleNamespace(turn=turn, conversation=SimpleNamespace())
    runner._peek_session_state = lambda _: state
    runner._persist_active_agents = lambda: None
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = OrderedDict(synthetic=(agent, 'sig'))
    runner._running_agent_ids = lambda: {id(turn.agent)} if turn.agent else set()
    runner._spawn_release_thread = lambda fn, args, *_, **kwargs: fn(*args)
    runner._evict_cached_agent('synthetic')
    assert runner._agent_cache['synthetic'][0] is agent
    client.close.assert_not_called()
    runner._release_running_agent_state('synthetic')
    assert not runner._agent_cache
    client.close.assert_called_once()
    assert agent._codex_session is None
