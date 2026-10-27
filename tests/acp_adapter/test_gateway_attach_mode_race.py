"""A first attach's reply shows the edit policy the owner enforces when it replies, not when it started.

Between the attach's ``session.resume`` and its reply, ``_session_fields`` waits on the model picker and the
provenance lookup. The pump already projects the session's events then, so another viewer's mode change in
that window must reach the reply: the editor never shows Default while the owner auto-allows edits."""
import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent

ASK, DONT_ASK = {'edit_approval_policy': 'ask'}, {'edit_approval_policy': 'session'}


@pytest.mark.asyncio
@pytest.mark.parametrize('attach', ['new', 'load'])
@pytest.mark.parametrize('lookup', ['model_state', 'provenance_meta'])
async def test_a_mode_change_while_the_attach_builds_its_reply_reaches_the_reply(attach, lookup, monkeypatch):
    monkeypatch.setattr('acp_adapter.catalog.logical_session_id', lambda db, held: 's')
    snapshot = {'session_id': 's', 'messages': [], 'prompts': [], 'pending': [], 'replay_epoch': 'e',
                'last_sequence': 10, 'execution_generation': 1, 'revision': 1, 'info': dict(ASK)}

    async def rpc(method, **params):
        return {'runtime.describe': {'session_create': {'sources': ['acp']}, 'capabilities': ['acp-editor-policy-v1']},
                'session.create': snapshot, 'session.resume': snapshot, 'session.info': dict(ASK),
                'config.set': {}, 'session.stats': {}}[method]

    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    gateway = type('Gateway', (), {})()
    gateway.events, gateway.rpc = asyncio.Queue(), rpc
    agent._gateway = gateway

    async def client():
        return gateway
    monkeypatch.setattr(agent, '_client', client)
    paused, release = threading.Event(), threading.Event()

    def paused_lookup(*args):  # the metadata lookup the attach is waiting on (off the loop)
        paused.set()
        release.wait(5)
    monkeypatch.setattr(f'acp_adapter.gateway_session_meta.{lookup}', paused_lookup)
    pump = asyncio.create_task(agent._events())
    attaching = asyncio.create_task(agent.new_session(cwd='/') if attach == 'new'
                                    else agent.load_session(cwd='/', session_id='s'))
    await asyncio.wait_for(asyncio.to_thread(paused.wait, 5), 5)
    # Another viewer picks Don't Ask: the owner's session.info reaches the pump while the attach is paused.
    await gateway.events.put({'params': {'session_id': 's', 'type': 'session.info', 'replay_epoch': 'e', 'seq': 11,
                                         'payload': dict(DONT_ASK)}})
    for _ in range(100):
        if agent._snapshots['s']['last_sequence'] == 11:
            break
        await asyncio.sleep(0)
    assert agent._snapshots['s']['last_sequence'] == 11, 'the pump never projected the change'
    release.set()
    response = await asyncio.wait_for(attaching, 5)
    pump.cancel()
    shown = [c.kwargs['update'].current_mode_id for c in agent._conn.session_update.await_args_list
             if getattr(c.kwargs['update'], 'session_update', None) == 'current_mode_update']
    assert response.modes.current_mode_id == 'dont_ask', (response.modes.current_mode_id, shown)
    assert agent._modes['s'] == 'dont_ask'
