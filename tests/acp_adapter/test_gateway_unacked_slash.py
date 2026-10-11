"""Slash work retires an un-acked prompt's retained identity: a later identical prompt is new work."""
import asyncio
from types import SimpleNamespace

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from gateway.session_controls import AuthorityConnection
from hermes_cli.gateway_client import GatewayClientError
from hermes_state_runtime import list_session_admissions
from tests.acp_adapter.test_gateway_submit_retry import Wire


class SlashWire(Wire):
    async def rpc(self, method, _timeout=None, **params):  # the client budget is never on the wire
        return await super().rpc(method, **params)


@pytest.mark.asyncio
async def test_repeated_prompt_after_slash_command_is_new_work(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    executed = []

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()['result'] = {'final_response': 'ACK_' + event.text, 'completed': True, 'messages': []}
        return 'ACK_' + event.text

    from gateway import run
    from gateway.session_local import create_local_session
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'acp': []}})
    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False, adapters={},
                             _handle_message=answer, _adapter_for_source=lambda source: None,
                             _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda **kwargs: ('frozen', {}))
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner')
    wire = SlashWire(asyncio.get_running_loop())
    wire.viewer = AuthorityConnection(authority, wire, {'user_id': 'editor', 'profile_id': 'p'})
    s = create_local_session(authority, wire.viewer.actor, {'request_id': 'editor', 'source': 'acp',
                                                           'cwd': str(tmp_path), 'model': 'frozen',
                                                           'toolsets': []}).session_id
    agent = GatewayACPAgent()
    agent._gateway = wire
    updates = []

    async def session_update(session_id, update):
        updates.append(update.content.text)
    agent._conn = SimpleNamespace(session_update=session_update)
    agent._snapshots[s] = await wire.rpc('session.resume', session_id=s)
    agent._event_task = asyncio.create_task(agent._events())
    try:
        with pytest.raises(GatewayClientError, match='outcome is unknown'):
            await agent.prompt([acp.text_block('yes')], s)
        async with asyncio.timeout(5):
            while 'ACK_yes' not in updates:
                await asyncio.sleep(0.01)
        # Read-only slash work between the lost-ack prompt and the deliberate repeat.
        await asyncio.wait_for(agent.prompt([acp.text_block('/compress --preview')], s), 5)
        response = await asyncio.wait_for(agent.prompt([acp.text_block('yes')], s), 5)
        assert response.stop_reason == 'end_turn'
        assert executed == ['yes', 'yes']
        assert len(list_session_admissions(authority.db, session_id=s, pending_only=False)) == 2
    finally:
        agent._event_task.cancel()
        await wire.viewer.close()
        store.close_all_db_handles()
