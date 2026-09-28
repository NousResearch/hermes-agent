"""``/new`` and ``/compress`` are session boundaries for memory providers, not resource evictions."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# session.create's resume path imports this lazily mid-request; importing it whole here keeps the
# conftest server-state fixture from meeting a half-initialized module at teardown.
import tui_gateway.server  # noqa: F401
from gateway.run import GatewayRunner


def _runner(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    import threading
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._peek_session_state = lambda key: None
    runner._running_agent_ids = lambda: set()
    # Run releases inline so the order of commit and release is observable.
    monkeypatch.setattr(runner, '_spawn_release_thread',
                        lambda target, args, name, *, inline_fallback, session_key=None: target(*args))
    return runner


def _agent(calls):
    agent = SimpleNamespace(_memory_manager=MagicMock(), _session_messages=[{'role': 'user', 'content': 'hi'}])
    agent.commit_memory_session = lambda msgs=None: calls.append(('commit', msgs))
    agent.release_clients = lambda: calls.append(('release', None))
    return agent


def test_boundary_evict_commits_the_transcript_before_releasing(monkeypatch):
    runner = _runner(monkeypatch)
    calls = []
    runner._agent_cache['route'] = (_agent(calls), 'sig')
    runner._evict_cached_agent_at_boundary('route')
    assert calls == [('commit', [{'role': 'user', 'content': 'hi'}]), ('release', None)]
    assert 'route' not in runner._agent_cache


def test_boundary_evict_commits_for_a_context_engine_without_a_provider(monkeypatch):
    """``commit_memory_session`` also ends the context engine's session; that engine runs without any
    external memory provider, so ``_memory_manager`` alone must not gate the boundary commit."""
    runner = _runner(monkeypatch)
    calls = []
    agent = _agent(calls)
    agent._memory_manager = None
    agent.context_compressor = MagicMock()
    runner._agent_cache['route'] = (agent, 'sig')
    runner._evict_cached_agent_at_boundary('route')
    assert calls == [('commit', [{'role': 'user', 'content': 'hi'}]), ('release', None)]


def test_evicts_with_nothing_to_notify_only_release(monkeypatch):
    """No provider and no engine: neither kind of evict has anything to commit. A resource evict with
    only an engine keeps the conversation open and does not end the engine's session either."""
    runner = _runner(monkeypatch)
    calls = []
    agent = _agent(calls)
    agent._memory_manager = None
    runner._agent_cache['route'] = (agent, 'sig')
    runner._evict_cached_agent_at_boundary('route')
    assert calls == [('release', None)]
    calls.clear()
    agent.context_compressor = MagicMock()
    runner._agent_cache['route'] = (agent, 'sig')
    runner._evict_cached_agent('route')
    assert calls == [('release', None)]


def test_plain_evict_still_skips_the_commit(monkeypatch):
    """``/model`` and rewind keep the conversation open; only the clients are recycled."""
    runner = _runner(monkeypatch)
    calls = []
    runner._agent_cache['route'] = (_agent(calls), 'sig')
    runner._evict_cached_agent('route')
    assert calls == [('release', None)]


@pytest.mark.asyncio
async def test_reset_and_compress_mutations_use_the_boundary_evict(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    import gateway.run as run

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda config: 'fixture')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    evicted = {'boundary': [], 'plain': []}
    runner = run.GatewayRunner.__new__(run.GatewayRunner)
    runner.adapters = {}
    runner.session_store = store
    runner._session_db = store._db
    runner._draining = False
    runner._evict_cached_agent = lambda route, **kw: evicted['plain'].append(route)
    runner._evict_cached_agent_at_boundary = lambda route: evicted['boundary'].append(route)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='test')
    authority._schedule = lambda ref: None
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'me'})
    created = await conn.dispatch({'id': 1, 'method': 'session.create', 'params': {'request_id': 'r', 'source': 'cli', 'cwd': str(tmp_path)}})
    assert 'result' in created, created
    sid = created['result']['session_id']
    route = authority.sessions[sid].route
    reset = await conn.dispatch({'id': 2, 'method': 'session.mutate', 'params': dict(
        session_id=sid, request_id='reset', expected_revision=0, expected_generation=0, operation='reset', payload={})})
    assert 'result' in reset, reset
    assert evicted['boundary'] == [route] and evicted['plain'] == []
    await conn.close()
    authority.db.close()
