"""An automation admission (no ACP prompt owns it) reaches an editor that asked for run boundaries as
ACP v2 ``state_update`` ``running`` ... ``idle``, carried by the ``_bb.dev/state_update`` extension."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent

CARD = {'display_kind': 'async_delegation_complete', 'display_text': 'Background delegation finished',
        'text': 'model-facing result'}


def agent(opted_in=True):
    agent = GatewayACPAgent()
    agent._snapshots['s'] = {}
    agent._gateway = AsyncMock()
    agent._conn = AsyncMock()
    agent._state_updates = opted_in
    return agent


def frame(kind, payload, aid='a'):
    return {'session_id': 's', 'type': kind, 'admission_id': aid, 'payload': payload}


def states(agent):
    """(runId, state, stopReason, error message) of every ``_bb.dev/state_update`` sent."""
    sent = []
    for call in agent._conn.ext_notification.await_args_list:
        method, params = call.args
        assert method == 'bb.dev/state_update' and params['sessionId'] == 's'
        update = params['update']
        assert update['sessionUpdate'] == 'state_update'
        sent.append((update['_meta']['runId'], update['state'], update.get('stopReason'),
                     (update.get('error') or {}).get('message')))
    return sent


async def wake(a, aid='a', outcome='completed', **complete):
    await a._project(frame('message.start', CARD, aid))
    for n in (1, 2):  # steps with model think time between them: no boundary of their own
        await a._project(frame('tool.start', {'tool_call_id': f'{aid}{n}', 'tool_name': 'terminal', 'args': {}}, aid))
        await a._project(frame('tool.complete', {'tool_call_id': f'{aid}{n}', 'tool_name': 'terminal',
                                                 'result': 'ok'}, aid))
    await a._project(frame('message.complete', {'text': 'Done.', 'outcome': outcome, **complete}, aid))


@pytest.mark.asyncio
async def test_one_wake_is_one_run_however_many_steps_it_takes():
    a = agent()
    await wake(a)
    assert states(a) == [('a', 'running', None, None), ('a', 'idle', 'end_turn', None)]
    assert a._automation_runs == set()


@pytest.mark.asyncio
async def test_each_wake_is_its_own_run():
    a = agent()
    await wake(a, 'a')
    await wake(a, 'b')
    assert [s[:3] for s in states(a)] == [('a', 'running', None), ('a', 'idle', 'end_turn'),
                                          ('b', 'running', None), ('b', 'idle', 'end_turn')]


@pytest.mark.asyncio
async def test_an_editor_that_did_not_ask_gets_no_extension_notification():
    a = agent(opted_in=False)
    await wake(a)
    a._conn.ext_notification.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_prompt_admission_has_no_card_and_no_run_boundary():
    a = agent()
    await a._project(frame('message.start', {}))
    await a._project(frame('message.complete', {'text': 'Hi.', 'outcome': 'completed'}))
    assert states(a) == []


@pytest.mark.asyncio
async def test_failed_and_cancelled_runs_end_with_their_outcome():
    a = agent()
    await wake(a, 'f', outcome='failed', failure_reason='provider 500')
    await wake(a, 'c', outcome='cancelled')
    assert states(a)[1] == ('f', 'idle', 'error', 'provider 500')
    assert states(a)[3] == ('c', 'idle', 'cancelled', None)


@pytest.mark.asyncio
async def test_a_newer_start_ends_a_run_whose_completion_never_came():
    a = agent()
    await a._project(frame('message.start', CARD, 'a'))
    await a._project(frame('message.start', CARD, 'b'))
    assert states(a) == [('a', 'running', None, None), ('a', 'idle', 'end_turn', None),
                         ('b', 'running', None, None)]


@pytest.mark.asyncio
async def test_duplicate_start_and_stale_complete_send_nothing_more():
    a = agent()
    await a._project(frame('message.start', CARD))
    await a._project(frame('message.start', CARD))
    await a._project(frame('message.complete', {'text': '', 'outcome': 'completed'}))
    await a._project(frame('message.complete', {'text': '', 'outcome': 'completed'}, 'never-started'))
    assert states(a) == [('a', 'running', None, None), ('a', 'idle', 'end_turn', None)]


@pytest.mark.asyncio
async def test_a_lost_link_ends_the_run_as_unknown_and_a_still_running_one_reopens_on_reattach():
    a = agent()
    await a._project(frame('message.start', CARD, 'a'))
    await a._project(frame('message.start', CARD, 'b'))  # 'a' superseded; 'b' open when the link drops
    a._event_task = a._connection = None
    await a._teardown()
    assert states(a)[-1][:3] == ('b', 'idle', 'error')
    gateway = AsyncMock()
    gateway.rpc.return_value = {'pending': [{'admission_id': 'b', 'status': 'started'},
                                            {'admission_id': 'z', 'status': 'started'}]}
    await a._reattach(gateway)
    assert states(a)[-1] == ('b', 'running', None, None)
    assert a._automation_runs == {('s', 'b')} and a._detached_runs == set()


@pytest.mark.asyncio
async def test_a_replay_gap_that_lost_the_run_ends_it_as_unknown():
    a = agent()
    a._snapshots['s'] = {'pending': []}  # ended in the hole
    await a._project(frame('message.start', CARD, 'a'))
    a._gateway.rpc.return_value = {'status': 'terminal', 'outcome': 'completed'}  # unreadable: no result
    await a._recover('s', 'a')
    assert states(a)[-1][:3] == ('a', 'idle', 'error')


@pytest.mark.asyncio
async def test_a_replay_gap_whose_receipt_settles_the_run_ends_it_normally():
    a = agent()
    a._snapshots['s'] = {'pending': []}
    await a._project(frame('message.start', CARD, 'a'))
    a._gateway.rpc.return_value = {'status': 'terminal', 'outcome': 'completed', 'admission_id': 'a',
                                   'result': {'final_response': 'Done.'}}
    await a._recover('s', 'a')
    assert states(a)[-1] == ('a', 'idle', 'end_turn', None)


@pytest.mark.parametrize('meta, expected', [
    ({'bb.dev': {'stateUpdate': 1}}, True), ({'bb.dev': {'stateUpdate': 2}}, True),
    ({'bb.dev': {'stateUpdate': True}}, False), ({'bb.dev': {}}, False), ({'stateUpdate': 1}, False), (None, False)])
def test_the_editor_opts_in_through_initialize_client_capabilities(meta, expected):
    a = GatewayACPAgent()
    caps = {'fs': {}} if meta is None else {'_meta': meta}
    a._observe_client_message(SimpleNamespace(direction='incoming', message={
        'method': 'initialize', 'params': {'protocolVersion': 1, 'clientCapabilities': caps}}))
    assert a._state_updates is expected


@pytest.mark.asyncio
async def test_stop_with_no_prompt_running_cancels_the_background_run_on_screen():
    a = agent()
    a._client = AsyncMock(return_value=a._gateway)
    await a._project(frame('message.start', CARD, 'wake'))
    await a.cancel('s')
    a._gateway.rpc.assert_awaited_once_with('prompt.cancel', session_id='s', admission_id='wake')


@pytest.mark.asyncio
async def test_stop_with_its_own_prompt_running_cancels_only_that_prompt():
    a = agent()
    a._client = AsyncMock(return_value=a._gateway)
    a._admissions['s'] = 'mine'
    await a._project(frame('message.start', CARD, 'wake'))
    await a.cancel('s')
    a._gateway.rpc.assert_awaited_once_with('prompt.cancel', session_id='s', admission_id='mine')


@pytest.mark.asyncio
async def test_stop_with_nothing_running_sends_nothing():
    a = agent()
    a._client = AsyncMock(return_value=a._gateway)
    await a.cancel('s')
    a._gateway.rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_dead_event_stream_ends_the_open_run_at_once_not_at_the_next_editor_call():
    a = agent()
    await a._project(frame('message.start', CARD, 'wake'))
    a._gateway.events = asyncio.Queue()
    a._gateway.events.put_nowait(ConnectionResetError('gateway gone'))
    await a._events()
    assert states(a)[-1][:3] == ('wake', 'idle', 'error')
    assert a._detached_runs == {('s', 'wake')}
