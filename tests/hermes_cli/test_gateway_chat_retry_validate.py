"""Classic /retry validates the target turn before rewinding it (dokterdok N92): real authority,
SQLite and AuthorityConnection.dispatch; an image turn is refused with nothing archived, a text-parts
turn is resubmitted as its lossless text."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class _Bridge:
    def __init__(self, connection):
        self.connection, self.events = connection, asyncio.Queue()

    async def rpc(self, method, **params):
        from hermes_cli.gateway_client import GatewayClientError
        reply = await self.connection.dispatch({'id': 1, 'method': method, 'params': params})
        if 'error' in reply:
            raise GatewayClientError(reply['error']['message'])
        return reply['result']


IMAGE = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}


@pytest.mark.asyncio
@pytest.mark.parametrize('content, submitted', [
    ([{'type': 'text', 'text': 'what is '}, {'type': 'text', 'text': 'this?'}], ['what is this?']),
    ([{'type': 'text', 'text': 'what is this?'}, IMAGE], None),
])
async def test_retry_refuses_an_unresendable_turn_before_rewinding(tmp_path, monkeypatch, content, submitted):
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
        db.append_message(ref.session_id, 'user', content)
        db.append_message(ref.session_id, 'assistant', 'earlier answer')
        view = SimpleNamespace(client=_Bridge(connection), session_id=ref.session_id,
                               mutations=PreparedMutations(), submit=AsyncMock(), generation=0)
        active = lambda: len(db.get_messages(ref.session_id))
        if submitted is None:
            with pytest.raises(GatewayClientError, match=r'^/retry: retry does not support media'):
                await run_command(view, '/retry', '')
            assert active() == 2 and not view.submit.called
        else:
            await run_command(view, '/retry', '')
            assert active() == 0
            assert [call.args[0] for call in view.submit.await_args_list] == submitted
    finally:
        store.close_all_db_handles()
