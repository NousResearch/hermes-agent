"""Owner refusals must survive the actual ACP JSON-RPC serializer."""
import asyncio
from contextlib import suppress
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['queue_full', 'unknown_execution', 'session_busy', 'permission_denied', 'storage_unavailable'])
async def test_prompt_refusal_reaches_editor_with_reason_and_owner_data(reason):
    agent = GatewayACPAgent()
    agent._snapshots['s'] = {}
    agent._gateway = SimpleNamespace(rpc=AsyncMock(side_effect=GatewayRPCError(
        reason, -32001, {'reason': reason, 'limit': 32})))
    agent._client = AsyncMock(return_value=agent._gateway)
    agent._prepare_submit = AsyncMock(return_value=('fingerprint', 'input', ('work', [])))
    agent._run_slash = AsyncMock(return_value=(False, 'work'))

    async def serve(reader, writer):
        protocol = acp.AgentSideConnection(agent, writer, reader, listening=False,
                                           use_unstable_protocol=True)
        try:
            await protocol.listen()
        finally:
            writer.transport.abort()
            with suppress(TimeoutError):
                await asyncio.wait_for(protocol.close(), 2)

    server = await asyncio.start_server(serve, '127.0.0.1', 0)
    reader, writer = await asyncio.open_connection('127.0.0.1', server.sockets[0].getsockname()[1])
    try:
        writer.write((json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'session/prompt',
                                 'params': {'sessionId': 's', 'prompt': [{'type': 'text', 'text': 'work'}]}}) + '\n').encode())
        await writer.drain()
        response = json.loads(await asyncio.wait_for(reader.readline(), 5))
        error = response['error']
        assert error['message'] != 'Internal error'
        assert reason in error['message']
        assert error['data']['reason'] == reason
        assert error['data']['limit'] == 32
        assert error['data']['gatewayCode'] == -32001
        assert error['data']['sessionId'] == 's'
        if reason == 'queue_full':
            assert 'queue' in error['data']['recovery']
        elif reason == 'unknown_execution':
            assert 'do not resend' in error['data']['recovery']
        assert agent._unacked == {}  # a definite refusal is not an ambiguous admission
    finally:
        writer.close()
        with suppress(ConnectionError):
            await writer.wait_closed()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize('method,kwargs,source', [
    ('cancel', {'session_id': 's'}, 'cancel'),
    ('load_session', {'session_id': 's', 'cwd': '.'}, 'client'),
    ('resume_session', {'session_id': 's', 'cwd': '.'}, 'client'),
    ('fork_session', {'session_id': 's', 'cwd': '.'}, 'client'),
    ('set_session_model', {'session_id': 's', 'model_id': 'm'}, 'client'),
    ('set_session_mode', {'session_id': 's', 'mode_id': 'default'}, 'mutation'),
    ('set_config_option', {'session_id': 's', 'config_id': 'edit_approval_policy', 'value': 'ask'}, 'mutation'),
    ('new_session', {'cwd': '.'}, 'client'),
])
async def test_gateway_entry_points_translate_expected_refusals(method, kwargs, source):
    agent = GatewayACPAgent()
    agent._snapshots['s'] = {}
    agent._admissions['s'] = 'admission'
    agent._conn = SimpleNamespace(session_update=AsyncMock())
    error = GatewayRPCError('storage_unavailable', -32001, {'reason': 'storage_unavailable'})
    if source == 'cancel':
        agent._cancel_admission = AsyncMock(side_effect=error)
    elif source == 'mutation':
        agent._mutations.apply = AsyncMock(side_effect=error)
        agent._client = AsyncMock(return_value=object())
    else:
        agent._client = AsyncMock(side_effect=error)
    with pytest.raises(acp.RequestError) as caught:
        await getattr(agent, method)(**kwargs)
    assert str(caught.value) == 'storage_unavailable'
    assert caught.value.data['reason'] == 'storage_unavailable'
    assert isinstance(caught.value, GatewayClientError)
    if method == 'cancel':
        agent._conn.session_update.assert_awaited_once()


@pytest.mark.asyncio
async def test_cached_unknown_and_missing_session_refusals_keep_recovery():
    agent = GatewayACPAgent()
    agent._snapshots['s'] = {'pending': [{'admission_id': 'lost', 'status': 'unknown'}]}
    agent._refresh_unknown_for_prompt = AsyncMock(return_value=False)
    with pytest.raises(acp.RequestError) as caught:
        await agent.prompt([acp.text_block('new work')], 's')
    assert caught.value.data['reason'] == 'unknown_execution'
    assert '/discard lost' in caught.value.data['recovery']
    assert 'nothing was submitted' in str(caught.value)
    with pytest.raises(acp.RequestError) as caught:
        await agent.prompt([acp.text_block('new work')], 'missing')
    assert caught.value.code == -32002
    assert caught.value.data['reason'] == 'not_found'


@pytest.mark.asyncio
async def test_transport_failure_is_not_reclassified_as_definite_owner_refusal():
    agent = GatewayACPAgent()
    agent._client = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(acp.RequestError) as caught:
        await agent.new_session('.')
    assert caught.value.data['reason'] == 'gateway_timeout'
    assert not isinstance(caught.value, GatewayRPCError)
    agent._client = AsyncMock(side_effect=RuntimeError('programming bug'))
    with pytest.raises(RuntimeError, match='programming bug'):
        await agent.new_session('.')


def test_owner_data_remains_bounded_and_local_reason_comparisons_survive():
    from acp_adapter.gateway_errors import request_error
    error = request_error(GatewayRPCError('queue_full', -32000, {'limit': 32, 'nested': {'secret': 'no'}}), 's')
    assert isinstance(error, GatewayRPCError)
    assert str(error) == 'queue_full'
    assert error.data['limit'] == 32 and 'nested' not in error.data
    assert error.data['gatewayCode'] == -32000
    assert error.to_error_obj()['code'] != -32000  # ACP reserves this for authentication required
