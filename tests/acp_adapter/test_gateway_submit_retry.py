"""An editor retry of a prompt whose submit ack was lost reuses its identity: one admission, one turn."""
import asyncio
from types import SimpleNamespace

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_authority import LiveSession, initialize_session_authority
from gateway.session_controls import AuthorityConnection
from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
from hermes_state_runtime import list_session_admissions


class Wire:
    """The ACP agent's gateway client over a real AuthorityConnection; can lose one submit's ack."""

    def __init__(self, loop):
        self.loop, self.events, self.serial, self.viewer, self.lose_ack = loop, asyncio.Queue(), 0, None, True

    def write(self, frame):  # the authority's transport (may be called off-loop)
        self.loop.call_soon_threadsafe(self.events.put_nowait, frame)
        return True

    async def rpc(self, method, **params):
        self.serial += 1
        reply = await self.viewer.dispatch({'id': self.serial, 'method': method, 'params': params})
        if 'error' in reply:
            raise GatewayRPCError(reply['error'].get('data', {}).get('reason', 'request_failed'))
        if method == 'prompt.submit' and self.lose_ack:
            self.lose_ack = False
            raise GatewayClientError('Gateway disconnected; turn outcome is unknown.')
        return reply['result']


@pytest.mark.asyncio
@pytest.mark.parametrize('settled_before_retry', [False, True])
async def test_retry_after_lost_submit_ack_is_one_admission(tmp_path, monkeypatch, settled_before_retry):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    executed = []

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()['result'] = {'final_response': 'ACK_' + event.text, 'completed': True, 'messages': []}
        return 'ACK_' + event.text

    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    store._db.create_session('s', source='telegram')
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner')
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='retry'), 's')
    wire = Wire(asyncio.get_running_loop())
    wire.viewer = AuthorityConnection(authority, wire, {'user_id': 'editor'})
    agent = GatewayACPAgent()
    agent._gateway = wire
    agent._conn = SimpleNamespace(updates=[])

    async def session_update(session_id, update):
        agent._conn.updates.append(update.content.text)
    agent._conn.session_update = session_update
    agent._snapshots['s'] = await wire.rpc('session.resume', session_id='s')
    agent._event_task = asyncio.create_task(agent._events())
    schedule = authority._schedule
    try:
        if not settled_before_retry:
            authority._schedule = lambda ref: None  # the lost turn is still queued at retry time
        with pytest.raises(GatewayClientError, match='outcome is unknown'):
            await agent.prompt([acp.text_block('same words')], 's')
        authority._schedule = schedule
        if settled_before_retry:
            async with asyncio.timeout(5):
                while 'ACK_same words' not in agent._conn.updates:
                    await asyncio.sleep(0.01)
        response = await asyncio.wait_for(agent.prompt([acp.text_block('same words')], 's'), 5)
        assert response.stop_reason == 'end_turn'
        rows = list_session_admissions(authority.db, session_id='s', pending_only=False)
        assert len(rows) == 1 and executed == ['same words']
        assert agent._conn.updates.count('ACK_same words') == 1
        # A different prompt is new work with a fresh identity.
        await asyncio.wait_for(agent.prompt([acp.text_block('other words')], 's'), 5)
        assert len(list_session_admissions(authority.db, session_id='s', pending_only=False)) == 2
    finally:
        agent._event_task.cancel()
        await wire.viewer.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_repeated_prompt_after_other_work_is_new_work(tmp_path, monkeypatch):
    """The retained identity of an un-acked prompt is only for the editor's retry (that session's
    next prompt): once other work was sent, the same words are a new turn, not a silent replay."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    executed = []

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()['result'] = {'final_response': 'ACK_' + event.text, 'completed': True, 'messages': []}
        return 'ACK_' + event.text

    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    store._db.create_session('s', source='telegram')
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner')
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='repeat'), 's')
    wire = Wire(asyncio.get_running_loop())
    wire.viewer = AuthorityConnection(authority, wire, {'user_id': 'editor'})
    agent = GatewayACPAgent()
    agent._gateway = wire
    agent._conn = SimpleNamespace(session_update=lambda **kwargs: asyncio.sleep(0))
    agent._snapshots['s'] = await wire.rpc('session.resume', session_id='s')
    agent._event_task = asyncio.create_task(agent._events())
    try:
        with pytest.raises(GatewayClientError, match='outcome is unknown'):
            await agent.prompt([acp.text_block('yes')], 's')
        for words in ('other words', 'yes'):
            response = await asyncio.wait_for(agent.prompt([acp.text_block(words)], 's'), 5)
            assert response.stop_reason == 'end_turn'
        assert executed == ['yes', 'other words', 'yes']
        assert len(list_session_admissions(authority.db, session_id='s', pending_only=False)) == 3
    finally:
        agent._event_task.cancel()
        await wire.viewer.close()
        store.close_all_db_handles()
