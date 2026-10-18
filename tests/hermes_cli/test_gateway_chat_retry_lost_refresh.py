"""Classic /retry keeps its prepared rewind until the post-rewind refresh lands: real authority,
SQLite and AuthorityConnection.dispatch; the one session.resume after the rewind is dropped, so the
user's repeat must replay the same rewind (one turn removed, the second question resent), never
re-derive a target from the already-shortened transcript."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class _DropRefreshOnce:
    def __init__(self, connection):
        self.connection, self.events = connection, asyncio.Queue()
        self.mutated, self.dropped = False, False

    async def rpc(self, method, **params):
        from hermes_cli.gateway_client import GatewayClientError
        if method == 'session.resume' and self.mutated and not self.dropped:
            self.dropped = True
            raise GatewayClientError('connection lost')
        reply = await self.connection.dispatch({'id': 1, 'method': method, 'params': params})
        if 'error' in reply:
            raise GatewayClientError(reply['error']['message'])
        if method == 'session.mutate':
            self.mutated = True
        return reply['result']


@pytest.mark.asyncio
async def test_retry_after_a_lost_refresh_replays_the_same_rewind(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_cli.gateway_chat_commands import run_command
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_mutations import PreparedMutations

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='owner')
    db = authority.db
    try:
        connection = AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': 'fixture'})
        ref = create_local_session(authority, connection.actor, {
            'request_id': 'cli', 'source': 'cli', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        for role, text in [('user', 'first question'), ('assistant', 'first answer'),
                           ('user', 'second question'), ('assistant', 'second answer')]:
            db.append_message(ref.session_id, role, text)
        client = _DropRefreshOnce(connection)
        view = SimpleNamespace(client=client, session_id=ref.session_id,
                               mutations=PreparedMutations(), submit=AsyncMock(), generation=0)
        with pytest.raises(GatewayClientError, match='connection lost'):
            await run_command(view, '/retry', '')
        await run_command(view, '/retry', '')
        remaining = [row['content'] for row in db.get_messages(ref.session_id)]
        assert remaining == ['first question', 'first answer']
        assert [call.args[0] for call in view.submit.await_args_list] == ['second question']
    finally:
        store.close_all_db_handles()
