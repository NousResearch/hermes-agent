"""The gateway's masked prompts (sudo, a skill's credential, the browser vault's panels) reach an ACP
editor that opts in through ``_hermes/masked_prompt``, and its answer goes back as the gateway's
``ValueResult`` response frame. An editor that does not opt in is sent nothing, as before."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from acp_adapter.masked_prompts import (
    CANCEL_METHOD, METHOD, answer, build_request, client_supports_masked_prompts,
)


def agent(opted_in=True):
    agent = GatewayACPAgent()
    agent._snapshots['s'] = {'last_sequence': 7, 'replay_epoch': 'e'}
    agent._gateway = MagicMock()
    agent._gateway.websocket.send = AsyncMock()
    agent._conn = MagicMock()
    agent._conn._conn.send_notification = AsyncMock()
    agent._masked_prompts = opted_in
    return agent


def srq(method, **params):
    return {'jsonrpc': '2.0', 'id': 'srq-0123456789ab', 'method': method, 'params': {'session_id': 's', **params}}


def replies(agent):
    return [json.loads(call.args[0]) for call in agent._gateway.websocket.send.await_args_list]


async def settle(agent):
    tasks = [task for _sid, task in agent._masked.values()]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize('params, expected', [
    ({'clientCapabilities': {'_meta': {'hermes': {'maskedPrompts': True}}}}, True),
    ({'capabilities': {'_meta': {'hermes': {'maskedPrompts': True}}}}, True),
    ({'clientCapabilities': {'_meta': {'hermes': {'maskedPrompts': 'yes'}}}}, False),
    ({'clientCapabilities': {'elicitation': {'form': {}}}}, False),
    ({}, False), (None, False),
])
def test_only_an_explicit_opt_in_counts(params, expected):
    assert client_supports_masked_prompts(params) is expected


def test_initialize_records_the_opt_in():
    editor = GatewayACPAgent()
    event = MagicMock(direction='incoming', message={'method': 'initialize', 'params': {
        'clientCapabilities': {'_meta': {'hermes': {'maskedPrompts': True}}}}})
    editor._observe_client_message(event)
    assert editor._masked_prompts is True


@pytest.mark.asyncio
async def test_a_sudo_prompt_reaches_the_editor_and_its_answer_the_gateway():
    editor = agent()
    editor._conn._conn.send_request = AsyncMock(return_value={'action': 'accept', 'values': {'value': 'hunter2'}})
    assert editor._server_request(srq('sudo', command='sudo apt update'))
    await settle(editor)
    method, params = editor._conn._conn.send_request.await_args.args
    assert method == METHOD
    assert params['sessionId'] == 's' and params['promptId'] == 'srq-0123456789ab' and params['kind'] == 'sudo'
    assert 'sudo apt update' in params['message']
    assert params['fields'] == [{'name': 'value', 'label': 'Password', 'masked': True}]
    assert 'session_id' not in params['details']
    assert replies(editor) == [{'jsonrpc': '2.0', 'id': 'srq-0123456789ab', 'result': {'value': 'hunter2'}}]
    assert editor._masked == {}


@pytest.mark.asyncio
async def test_a_request_frame_never_moves_the_replay_cursor():
    """An ``srq-`` frame has no ``seq``: before this it reset ``last_sequence`` to 0, and the next
    event was taken for new or old by the wrong cursor."""
    from tests.waiting import async_wait_until
    editor = agent(opted_in=False)
    editor._gateway.events = asyncio.Queue()
    editor._project = AsyncMock()
    await editor._gateway.events.put(srq('secret', env_var='API_TOKEN', prompt='Enter it'))
    event = {'session_id': 's', 'type': 'message.delta', 'seq': 8, 'replay_epoch': 'e', 'payload': {}}
    await editor._gateway.events.put({'jsonrpc': '2.0', 'method': 'event', 'params': event})
    pump = asyncio.create_task(editor._events())
    await async_wait_until(lambda: editor._project.await_count, what='the event after the request')
    pump.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pump
    editor._project.assert_awaited_once_with(event)
    assert editor._snapshots['s']['last_sequence'] == 8 and editor._snapshots['s']['replay_epoch'] == 'e'


@pytest.mark.asyncio
async def test_an_editor_that_did_not_opt_in_is_sent_nothing():
    editor = agent(opted_in=False)
    editor._conn._conn.send_request = AsyncMock()
    assert editor._server_request(srq('sudo', command='sudo ls'))
    editor._conn._conn.send_request.assert_not_called()
    editor._gateway.websocket.send.assert_not_called()


@pytest.mark.asyncio
async def test_other_sessions_and_other_requests_are_left_alone():
    editor = agent()
    editor._conn._conn.send_request = AsyncMock()
    other = srq('sudo')
    other['params']['session_id'] = 'elsewhere'
    assert editor._server_request(other)
    assert editor._server_request(srq('clarify', question='?'))
    assert not editor._server_request({'jsonrpc': '2.0', 'method': 'event', 'params': {'session_id': 's'}})
    editor._conn._conn.send_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [{'action': 'decline'}, {'action': 'cancel'}, {'action': 'accept'}, None])
async def test_a_declined_prompt_is_the_skip(response):
    editor = agent()
    editor._conn._conn.send_request = AsyncMock(return_value=response)
    editor._server_request(srq('vault.code', site='bank', hint='Check your phone'))
    await settle(editor)
    assert replies(editor)[0]['result'] == {'value': ''}


@pytest.mark.asyncio
async def test_an_editor_error_is_the_skip_not_a_hang():
    editor = agent()
    editor._conn._conn.send_request = AsyncMock(side_effect=RuntimeError('-32601'))
    editor._server_request(srq('vault.unlock_prompt', backend='bitwarden', display_name='Bitwarden'))
    await settle(editor)
    assert replies(editor)[0]['result'] == {'value': ''}


@pytest.mark.asyncio
async def test_a_replayed_request_is_not_asked_twice():
    editor = agent()
    gate = asyncio.Event()

    async def ask(*_args):
        await gate.wait()
        return {'action': 'accept', 'values': {'value': 'x'}}
    editor._conn._conn.send_request = AsyncMock(side_effect=ask)
    editor._server_request(srq('sudo'))
    editor._server_request(srq('sudo'))
    await asyncio.sleep(0)
    gate.set()
    await settle(editor)
    assert editor._conn._conn.send_request.await_count == 1
    assert len(replies(editor)) == 1


@pytest.mark.asyncio
async def test_a_request_the_gateway_withdrew_closes_on_the_editor():
    editor = agent()

    async def never(*_args):
        await asyncio.Event().wait()
    editor._conn._conn.send_request = AsyncMock(side_effect=never)
    editor._server_request(srq('sudo'))
    await asyncio.sleep(0)
    await editor._project({'session_id': 's', 'type': 'request.cancel',
                           'payload': {'id': 'srq-0123456789ab', 'method': 'sudo', 'reason': 'timeout'}})
    await asyncio.sleep(0)
    editor._conn._conn.send_notification.assert_awaited_once_with(
        CANCEL_METHOD, {'sessionId': 's', 'promptId': 'srq-0123456789ab', 'reason': 'timeout'})
    editor._gateway.websocket.send.assert_not_called()
    assert editor._masked == {}


@pytest.mark.parametrize('kind, params', [
    ('sudo', {}), ('secret', {'env_var': 'TOKEN', 'prompt': 'Enter TOKEN', 'metadata': {'skill_name': 'gh'}}),
    ('vault.unlock_prompt', {'backend': 'bw', 'display_name': 'Bitwarden'}),
    ('vault.save_login', {'origin': 'https://a.test', 'site': 'a.test'}), ('vault.code', {}),
])
def test_every_kind_builds_a_card(kind, params):
    request = build_request('s', 'srq-1', kind, {'session_id': 's', **params})
    assert request['title'] and request['message'] and request['fields']
    assert all(field['name'] and field['label'] for field in request['fields'])


def test_save_login_goes_back_as_the_cards_json_string():
    value = answer('vault.save_login', {'action': 'accept', 'values': {'identifier': 'me', 'password': 'pw'}})
    assert json.loads(value) == {'identifier': 'me', 'password': 'pw'}
    assert answer('vault.save_login', {'action': 'accept', 'values': {'identifier': 'me', 'password': ''}}) == ''


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        build_request('s', 'srq-1', 'clarify', {})
