"""A compressed editor chat reopens under its logical id (real authority + SQLite, real dispatch).

Compression advances the physical transcript to a new row; the authority's policy, FIFO and live
session stay keyed by the logical (root) id. The picker and an editor-held tip id must both land
on that logical id before the first ``session.info``, never on the tip's (absent) policy.
"""
import asyncio
from types import SimpleNamespace

import pytest


class _Bridge:
    """``GatewayClient.rpc`` over a real ``AuthorityConnection.dispatch`` (no socket)."""

    def __init__(self, connection):
        self.connection, self.events = connection, asyncio.Queue()

    async def rpc(self, method, **params):
        from hermes_cli.gateway_client import GatewayClientError
        reply = await self.connection.dispatch({'id': 1, 'method': method, 'params': params})
        if 'error' in reply:
            raise GatewayClientError(reply['error']['message'])
        return reply['result']


class _Editor:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(session_id)


@pytest.mark.asyncio
async def test_compressed_acp_chat_lists_and_reopens_under_its_logical_id(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from acp_adapter.gateway_server import GatewayACPAgent

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'acp': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='owner')
    db = authority.db
    try:
        connection = AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': 'fixture'})
        ref = create_local_session(authority, connection.actor, {
            'request_id': 'editor', 'source': 'acp', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        db.append_message(ref.session_id, 'user', 'before compression')
        assert db.try_acquire_compression_lock(ref.session_id, 'fixture')
        db.publish_compression_child(
            parent_session_id=ref.session_id, child_session_id='compressed-tip', source='acp',
            messages=[{'role': 'user', 'content': 'summary'}, {'role': 'assistant', 'content': 'retained'}],
            compression_lock_holder='fixture')
        db.release_compression_lock(ref.session_id, 'fixture')

        agent = GatewayACPAgent()
        agent._home = tmp_path
        agent._gateway, agent._conn = _Bridge(connection), _Editor()
        listed = [row.session_id for row in (await agent.list_sessions(cwd=str(tmp_path))).sessions]
        assert listed == [ref.session_id], 'the picker must offer the logical id, not the physical tip'
        await agent.load_session(cwd=str(tmp_path), session_id=listed[0], mcp_servers=[])
        assert list(agent._snapshots) == [ref.session_id]

        # An editor that saved the tip id before this fix still reopens the same conversation,
        # and every update it sees stays addressed to the id it holds.
        held = GatewayACPAgent()
        held._home = tmp_path
        held._gateway, held._conn = _Bridge(connection), _Editor()
        await held.load_session(cwd=str(tmp_path), session_id='compressed-tip', mcp_servers=[])
        assert list(held._snapshots) == [ref.session_id]
        assert set(held._conn.updates) == {'compressed-tip'}
        await held._project({'session_id': ref.session_id, 'type': 'message.delta', 'admission_id': 'a',
                             'payload': {'text': 'live'}})
        assert held._conn.updates[-1] == 'compressed-tip'
        await held.cancel('compressed-tip')  # resolves to the logical owner instead of not_found
    finally:
        store.close_all_db_handles()
