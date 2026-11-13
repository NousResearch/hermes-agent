"""Guidance failures and session aliases cannot alter the primary ACP contract."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from acp.schema import TextContentBlock
from acp.exceptions import RequestError

from acp_adapter.gateway_server import GatewayACPAgent
from agent.active_turn_guidance import CAPABILITY
from hermes_cli.gateway_client import GatewayClientError
from tests.acp_adapter.test_gateway_terminal_receipt_claim import _agent


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['snapshot', 'notification'])
async def test_start_snapshot_failure_preserves_committed_primary_result(failure):
    agent = _agent('COMMITTED PRIMARY')
    agent._guidance.negotiated = True
    original_rpc = agent._gateway.rpc.side_effect

    async def rpc(method, **params):
        if method == 'session.guidance.turn':
            if failure == 'snapshot':
                raise GatewayClientError('auxiliary snapshot response lost')
            return {'state': 'started', 'runtimeInstanceId': 'owner', 'sessionId': 's', 'turnId': 'turn'}
        return await original_rpc(method, **params)

    agent._gateway.rpc.side_effect = rpc
    agent._conn.ext_notification.side_effect = BrokenPipeError('auxiliary notification failed')
    agent._usage['s'] = {'input_tokens': 27, 'output_tokens': 8, 'total_tokens': 35}
    result = await agent.prompt([TextContentBlock(type='text', text='hello')], 's',
                               **{CAPABILITY: {'promptId': 'host'}})
    assert result.stop_reason == 'end_turn'
    assert agent._conn.session_update.await_args.kwargs['update'].content.text == 'COMMITTED PRIMARY'
    assert result.usage.input_tokens == 27
    assert result.usage.output_tokens == 8
    calls = [c.args[0] for c in agent._gateway.rpc.await_args_list]
    assert calls.count('prompt.submit') == 1
    assert 'session.guidance.admit' not in calls
    assert result.field_meta is None
    if failure == 'snapshot':
        agent._conn.ext_notification.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['snapshot', 'notification'])
async def test_start_snapshot_failure_does_not_stop_primary_event_projection(failure):
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._gateway.events = asyncio.Queue()
    async def rpc(method, **params):
        if failure == 'snapshot':
            raise GatewayClientError('auxiliary snapshot response lost')
        return {'state': 'started', 'runtimeInstanceId': 'owner', 'sessionId': 's', 'turnId': 'turn'}
    agent._gateway.rpc.side_effect = rpc
    agent._conn = AsyncMock()
    agent._conn.ext_notification.side_effect = BrokenPipeError('auxiliary notification failed')
    agent._snapshots['s'] = {}
    agent._guidance.negotiated = True
    agent._guidance.track('s', 'a', 'host')
    projected = asyncio.Event()

    async def update(**kwargs):
        if getattr(kwargs['update'], 'session_update', None) == 'agent_message_chunk':
            assert kwargs['update'].content.text == 'PRIMARY'
            projected.set()

    agent._conn.session_update.side_effect = update
    pump = asyncio.create_task(agent._events())
    notice = asyncio.create_task(projected.wait())
    try:
        for seq, kind, payload in [(1, 'message.start', {}),
                                   (2, 'session.usage', {'input_tokens': 27, 'output_tokens': 8, 'total_tokens': 35}),
                                   (3, 'message.complete', {'outcome': 'completed', 'text': 'PRIMARY'})]:
            await agent._gateway.events.put({'params': {'session_id': 's', 'admission_id': 'a',
                'seq': seq, 'replay_epoch': 'epoch', 'type': kind, 'payload': payload}})
        await asyncio.wait([pump, notice], timeout=2, return_when=asyncio.FIRST_COMPLETED)
        assert agent._failure is None, f'auxiliary failure killed primary pump: {agent._failure}'
        assert projected.is_set()
        assert agent._terminals['a']['outcome'] == 'completed'
        assert agent._usage['s']['input_tokens'] == 27
        assert agent._usage['s']['output_tokens'] == 8
        assert all(c.args[0] == 'session.guidance.turn' for c in agent._gateway.rpc.await_args_list)
    finally:
        pump.cancel()
        notice.cancel()
        await asyncio.gather(pump, notice, return_exceptions=True)


@pytest.mark.asyncio
async def test_guidance_notification_uses_the_held_acp_session_alias():
    agent = GatewayACPAgent()
    agent._aliases['editor-tip'] = 'canonical-session'
    agent._snapshots['canonical-session'] = {}
    agent._gateway = AsyncMock()
    agent._conn = AsyncMock()
    agent._guidance.negotiated = True
    agent._guidance.track('canonical-session', 'a', 'host')
    agent._gateway.rpc.return_value = {'state': 'started', 'runtimeInstanceId': 'owner:epoch',
                                      'sessionId': 'canonical-session', 'turnId': 'turn'}
    await agent._guidance.started_turn('canonical-session', 'a')
    notification = agent._conn.ext_notification.await_args.args[1]
    assert notification['sessionId'] == 'editor-tip', notification


@pytest.mark.asyncio
async def test_guidance_request_resolves_the_held_alias_at_the_owner_boundary():
    agent = GatewayACPAgent()
    agent._aliases['editor-tip'] = 'canonical-session'
    agent._snapshots['canonical-session'] = {}
    agent._gateway = AsyncMock()
    agent._guidance.negotiated = True
    agent._gateway.rpc.return_value = {'sessionId': 'canonical-session', 'state': 'incorporated'}
    receipt = await agent._guidance.request('ai.nous/guidanceStatus', {
        'sessionId': 'editor-tip', 'runtimeInstanceId': 'owner:epoch',
        'expectedTurnId': 'turn', 'messageId': 'm'})
    assert agent._gateway.rpc.await_args.kwargs['session_id'] == 'canonical-session'
    assert receipt['sessionId'] == 'editor-tip'


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['read', 'notification', 'terminal'])
async def test_guidance_preserves_primary_cancellation(phase):
    agent = GatewayACPAgent()
    agent._gateway, agent._conn = AsyncMock(), AsyncMock()
    agent._guidance.negotiated = True
    agent._guidance.track('s', 'a', 'host')
    agent._gateway.rpc.return_value = {'state': 'started', 'runtimeInstanceId': 'owner',
                                      'sessionId': 's', 'turnId': 'turn'}
    if phase == 'notification':
        agent._conn.ext_notification.side_effect = asyncio.CancelledError
    else:
        agent._gateway.rpc.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        if phase == 'terminal':
            await agent._guidance.terminal('s', 'a')
        else:
            await agent._guidance.started_turn('s', 'a')


@pytest.mark.asyncio
@pytest.mark.parametrize('via_events', [False, True])
async def test_primary_cancellation_propagates_through_guidance_callers(via_events):
    agent = _agent('COMMITTED PRIMARY')
    agent._guidance.negotiated = True
    original_rpc = agent._gateway.rpc.side_effect
    async def rpc(method, **params):
        if method == 'session.guidance.turn':
            raise asyncio.CancelledError
        return await original_rpc(method, **params)
    agent._gateway.rpc.side_effect = rpc
    with pytest.raises(asyncio.CancelledError):
        if via_events:
            agent._guidance.track('s', 'a', 'host')
            agent._gateway.events = asyncio.Queue()
            await agent._gateway.events.put({'params': {'session_id': 's', 'admission_id': 'a',
                'type': 'message.start', 'seq': 1, 'replay_epoch': 'epoch', 'payload': {}}})
            await asyncio.wait_for(agent._events(), 2)
        else:
            await agent.prompt([TextContentBlock(type='text', text='hello')], 's',
                               **{CAPABILITY: {'promptId': 'host'}})
    assert agent._failure is None


@pytest.mark.asyncio
async def test_terminal_alias_translation_keeps_foreign_receipts_uncorrelated():
    agent = GatewayACPAgent()
    agent._aliases['editor-tip'] = 'canonical-session'
    agent._gateway = AsyncMock()
    snapshot = {'state': 'ended', 'sessionId': 'canonical-session', 'receipts': [
        {'sessionId': 'canonical-session', 'messageId': 'ours', 'state': 'incorporated'},
        {'sessionId': 'foreign', 'messageId': 'other', 'state': 'incorporated'}]}
    agent._gateway.rpc.return_value = snapshot
    agent._guidance.negotiated = True
    agent._guidance.track('canonical-session', 'a', 'host')
    result = (await agent._guidance.terminal('canonical-session', 'a'))[CAPABILITY]
    assert result['sessionId'] == 'editor-tip'
    assert result['receipts'][0]['sessionId'] == 'editor-tip'
    assert result['receipts'][1]['sessionId'] == 'foreign'
    assert snapshot['sessionId'] == 'canonical-session'
    assert snapshot['receipts'][0]['sessionId'] == 'canonical-session'


@pytest.mark.asyncio
async def test_guidance_alias_does_not_authorize_an_unheld_session():
    agent = GatewayACPAgent()
    agent._aliases['editor-tip'] = 'foreign'
    agent._snapshots['canonical-session'] = {}
    agent._gateway = AsyncMock()
    agent._guidance.negotiated = True
    with pytest.raises(RequestError):
        await agent._guidance.request('ai.nous/steer', {'sessionId': 'editor-tip', 'runtimeInstanceId': 'owner',
            'expectedTurnId': 'turn', 'messageId': 'm', 'text': 'must not be submitted'})
    agent._gateway.rpc.assert_not_awaited()
