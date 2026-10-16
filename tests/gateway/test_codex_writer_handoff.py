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

    def create(*, delayed=False):
        entered, proceed = threading.Event(), threading.Event()

        class Session(CodexAppServerSession):
            def close(self):
                if delayed and not self._closed:
                    entered.set()
                    assert proceed.wait(10), 'test never released the retirement gate'
                super().close()

        session = Session(cwd=str(tmp_path), codex_bin=str(script), codex_home=str(tmp_path),
                          resume_thread_id='synthetic-thread', model='synthetic-model')
        session.entered, session.proceed = entered, proceed
        sessions.append(session)
        assert session.ensure_started() == 'synthetic-thread'
        processes.append(session._client._proc)
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
        replacement = agent_shell(create())
        built.append(replacement)
        return replacement

    turn._build_fresh_agent = build
    route = {'model': 'synthetic-model', 'runtime': {}}
    resolve = lambda: turn._resolve_turn_agent(route, channel, '', 5, {}, {})
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
            assert not built
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
    from gateway.agent_cache_release import codex_release_barrier

    client = transport.CodexAppServerClient.__new__(transport.CodexAppServerClient)
    client._closed = False
    client._pending = {}
    client._pending_lock = threading.Lock()
    client._proc = Mock(stdin=None, pid=123456,
                        wait=Mock(side_effect=subprocess.TimeoutExpired('synthetic-codex', 1)))
    monkeypatch.setattr(transport, '_snapshot_descendants', lambda _: [])
    monkeypatch.setattr(transport, 'kill_process_tree', lambda _: None)
    session = CodexAppServerSession()
    session._client = client
    agent = agent_shell(session)
    runner = GatewayRunner.__new__(GatewayRunner)
    barrier = codex_release_barrier(runner)
    barrier.track('synthetic', agent)
    with pytest.raises(subprocess.TimeoutExpired):
        barrier.release(agent)
    with pytest.raises(subprocess.TimeoutExpired):
        barrier.wait('synthetic')
    agent.release_clients.assert_not_called()
