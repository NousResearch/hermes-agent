"""A fork's editor MCP discovery runs only after the branch's own fences pass (#134889 ACP-12).

Discovery launches every requested server, so a stale, busy or exact-retry ``branch`` must be
answered by the receipt, revision, generation and idle checks before any server starts.
Real authority and SQLite, real ``session.mutate`` dispatch; the discovery entry point is a spy.
"""
import asyncio
from types import SimpleNamespace

import pytest


class _Bridge:
    def __init__(self, connection):
        self.connection = connection

    async def rpc(self, method, **params):
        reply = await self.connection.dispatch({'id': 1, 'method': method, 'params': params})
        if 'error' in reply:
            return reply['error']['message']
        return reply['result']


@pytest.fixture
def owner(tmp_path, monkeypatch, server_spec):
    from gateway import run, session_local_mcp
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'acp': []}})
    discovered = []
    real = session_local_mcp.rebind_editor_mcp

    def spy(authority, sid, policy, servers):
        discovered.append(sid)
        return real(authority, sid, policy, servers)
    monkeypatch.setattr(session_local_mcp, 'rebind_editor_mcp', spy)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)

    built = []

    async def build():
        authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='owner')
        built.append(authority)
        connection = AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': 'fixture'})
        ref = create_local_session(authority, connection.actor, {
            'request_id': 'editor', 'source': 'acp', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        return authority, _Bridge(connection), ref.session_id
    try:
        yield build, discovered, {'mcp_servers': [server_spec]}
    finally:
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        for authority in built:
            for scope in list(getattr(authority, '_editor_mcp_held', {})):
                shutdown_mcp_servers(scope=scope)
        store.close_all_db_handles()


def _branch(sid, payload, **fence):
    return dict(session_id=sid, request_id='fork', operation='branch', payload=payload,
                expected_revision=fence.get('revision', 0), expected_generation=fence.get('generation', 0))


@pytest.mark.asyncio
async def test_stale_branch_starts_no_editor_server(owner):
    build, discovered, payload = owner
    _, client, sid = await build()
    assert await client.rpc('session.mutate', **_branch(sid, payload, revision=7)) == 'revision_conflict'
    assert await client.rpc('session.mutate', **_branch(sid, payload, generation=7)) == 'stale_generation'
    assert discovered == []


@pytest.mark.asyncio
async def test_busy_branch_starts_no_editor_server(owner):
    from hermes_state_runtime import admit_session_input
    build, discovered, payload = owner
    authority, client, sid = await build()
    # A queued input: the branch commit's own idle rule refuses it (require_idle).
    await asyncio.to_thread(admit_session_input, authority.db, epoch=authority.epoch, principal_id='human',
                            session_id=sid, request_id='queued', payload={'text': 'waiting'})
    row = authority.db.get_session(sid)
    fence = dict(revision=row['runtime_revision'], generation=row['runtime_generation'])
    assert await client.rpc('session.mutate', **_branch(sid, payload, **fence)) == 'session_busy'
    assert discovered == []


@pytest.mark.asyncio
async def test_exact_branch_retry_returns_the_receipt_without_discovery(owner):
    build, discovered, payload = owner
    _, client, sid = await build()
    first = await client.rpc('session.mutate', **_branch(sid, payload))
    assert isinstance(first, dict) and first['branched_session_id'], first
    assert discovered == [first['branched_session_id']]
    assert await client.rpc('session.mutate', **_branch(sid, payload)) == first
    assert discovered == [first['branched_session_id']], 'an exact retry must not rediscover'
