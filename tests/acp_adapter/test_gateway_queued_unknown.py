"""Editor prompts fail recoverably when their FIFO becomes unknown."""
import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, call

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('before_ack', [False, True])
async def test_editor_follower_reports_unknown_without_poisoning_other_sessions(before_ack):
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._snapshots['s'] = {}
    agent._snapshots['other'] = {}
    unknown = {'session_id': 's', 'type': 'session.info', 'payload': {'pending': [
        {'admission_id': 'head', 'status': 'unknown'}, {'admission_id': 'ours', 'status': 'queued'}]}}

    async def submit(method, **params):
        assert method == 'prompt.submit'
        if params['session_id'] == 'other':
            await agent._project({'session_id': 'other', 'type': 'message.complete',
                                  'admission_id': 'other-turn', 'payload': {'outcome': 'completed'}})
            return {'admission_id': 'other-turn'}
        if before_ack:
            await agent._project(unknown)
        return {'admission_id': 'ours'}

    agent._gateway.rpc.side_effect = submit
    submitted = asyncio.create_task(agent.prompt([acp.text_block('queued')], 's'))
    if not before_ack:
        await asyncio.sleep(0)
        await agent._project(unknown)
    with pytest.raises(GatewayClientError, match='unknown_execution: do not resend accepted input'):
        await asyncio.wait_for(submitted, 5)
    agent._gateway.rpc.assert_awaited_once()
    assert agent._failure is None
    assert (await agent.prompt([acp.text_block('independent')], 'other')).stop_reason == 'end_turn'


@pytest.mark.asyncio
async def test_editor_refuses_new_prompt_until_lost_turn_is_resolved():
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._snapshots['s'] = {'pending': [{'admission_id': 'lost', 'status': 'unknown'}]}
    agent._gateway.rpc.return_value = {'pending': [{'admission_id': 'lost', 'status': 'unknown'}]}
    with pytest.raises(GatewayClientError, match='unknown_execution'):
        await asyncio.wait_for(agent.prompt([acp.text_block('new input')], 's'), 5)
    agent._gateway.rpc.assert_awaited_once_with('session.resume', session_id='s')


@pytest.mark.asyncio
async def test_editor_rechecks_lost_turn_after_discard_in_another_viewer():
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    lost = [{'admission_id': 'lost', 'status': 'unknown'}]
    agent._snapshots['s'] = {'pending': lost, 'replay_epoch': 'owner', 'last_sequence': 1}
    owner_pending = list(lost)
    calls = []

    async def rpc(method, **params):
        calls.append(method)
        if method == 'session.resume':
            return {'pending': list(owner_pending), 'replay_epoch': 'owner', 'last_sequence': 2}
        assert method == 'prompt.submit' and params['session_id'] == 's'
        await agent._project({'session_id': 's', 'type': 'message.complete',
                              'admission_id': 'next', 'payload': {'outcome': 'completed'}})
        return {'admission_id': 'next'}

    agent._gateway.rpc.side_effect = rpc
    with pytest.raises(GatewayClientError, match='unknown_execution'):
        await agent.prompt([acp.text_block('still blocked')], 's')
    assert 'prompt.submit' not in calls

    owner_pending.clear()  # The classic chat acknowledged the lost turn with /discard.
    response = await asyncio.wait_for(agent.prompt([acp.text_block('next prompt')], 's'), 5)
    assert response.stop_reason == 'end_turn'
    assert calls.count('prompt.submit') == 1


@pytest.mark.asyncio
async def test_unknown_refresh_keeps_unread_reply_events_without_restoring_old_pending():
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._gateway.events = asyncio.Queue()
    agent._conn = AsyncMock()
    approval_seen, release_approval = asyncio.Event(), asyncio.Event()

    async def show_approval(_session_id, _prompt):
        approval_seen.set()
        await release_approval.wait()
        return True

    agent._answer_permission = show_approval
    agent._snapshots['s'] = {'pending': [{'admission_id': 'lost', 'status': 'unknown'}],
                             'replay_epoch': 'owner', 'last_sequence': 1}
    agent._gateway.rpc.return_value = {'pending': [], 'replay_epoch': 'owner', 'last_sequence': 4}

    await agent._refresh_unknown('s')
    assert not agent._has_unknown('s')
    # The event pump was busy while the owner discarded the head and ran a follower.
    # Its old pending event must not undo the refresh; its reply still reaches the editor.
    await agent._gateway.events.put({'params': {
        'session_id': 's', 'replay_epoch': 'owner', 'seq': 2, 'type': 'session.info',
        'payload': {'pending': [{'admission_id': 'lost', 'status': 'unknown'}]}}})
    await agent._gateway.events.put({'params': {
        'session_id': 's', 'replay_epoch': 'owner', 'seq': 3, 'type': 'approval.request',
        'payload': {'kind': 'approval', 'prompt_id': 'approval-1', 'execution_generation': 1}}})
    await agent._gateway.events.put({'params': {
        'session_id': 's', 'replay_epoch': 'owner', 'seq': 4, 'type': 'message.delta',
        'admission_id': 'follower', 'payload': {'text': 'Follower answer'}}})
    task = asyncio.create_task(agent._events())
    try:
        await async_wait_until(lambda: approval_seen.is_set() and agent._conn.session_update.await_count == 1,
                               what='the buffered approval and follower reply reached the editor')
        assert agent._conn.session_update.await_args.kwargs['update'].content.text == 'Follower answer'
        assert not agent._has_unknown('s')
    finally:
        release_approval.set()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await asyncio.gather(*agent._permissions.values(), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('newer_pending', [False, True])
async def test_unknown_refresh_orders_pending_state_independently_of_reply_events(newer_pending):
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._gateway.events = asyncio.Queue()
    agent._conn = AsyncMock()
    lost = [{'admission_id': 'lost', 'status': 'unknown'}]
    agent._snapshots['s'] = {'pending': lost, 'replay_epoch': 'owner', 'last_sequence': 1}
    entered, release = asyncio.Event(), asyncio.Event()

    async def rpc(method, **params):
        if method == 'session.resume':
            entered.set()
            await release.wait()
            return {'pending': [], 'replay_epoch': 'owner', 'last_sequence': 3}
        assert method == 'prompt.submit'
        await agent._project({'session_id': 's', 'type': 'message.complete',
                              'admission_id': 'next', 'payload': {'outcome': 'completed'}})
        return {'admission_id': 'next'}

    agent._gateway.rpc.side_effect = rpc
    pump = asyncio.create_task(agent._events())
    prompt = asyncio.create_task(agent.prompt([acp.text_block('next prompt')], 's'))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        # While the resume reply is assembled, the pump drains old pending state and a
        # follower's newer reply. Only a newer pending event can supersede the snapshot.
        await agent._gateway.events.put({'params': {
            'session_id': 's', 'replay_epoch': 'owner', 'seq': 2,
            'type': 'session.info', 'payload': {'pending': lost}}})
        await agent._gateway.events.put({'params': {
            'session_id': 's', 'replay_epoch': 'owner', 'seq': 4, 'type': 'message.delta',
            'admission_id': 'follower', 'payload': {'text': 'Follower answer'}}})
        if newer_pending:
            await agent._gateway.events.put({'params': {
                'session_id': 's', 'replay_epoch': 'owner', 'seq': 5, 'type': 'session.info',
                'payload': {'pending': [{'admission_id': 'follower', 'status': 'unknown'}]}}})
        await async_wait_until(lambda: agent._snapshots['s']['last_sequence'] == (5 if newer_pending else 4),
                               what='the follower events reached the editor before the resume response')
        assert agent._conn.session_update.await_args.kwargs['update'].content.text == 'Follower answer'
        release.set()
        if newer_pending:
            with pytest.raises(GatewayClientError, match='unknown_execution'):
                await asyncio.wait_for(prompt, 5)
        else:
            assert (await asyncio.wait_for(prompt, 5)).stop_reason == 'end_turn'
        methods = [call.args[0] for call in agent._gateway.rpc.await_args_list]
        assert methods.count('prompt.submit') == (0 if newer_pending else 1)
    finally:
        release.set()
        prompt.cancel()
        pump.cancel()
        await asyncio.gather(prompt, pump, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_during_unknown_refresh_cancels_the_unsent_prompt():
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._snapshots['s'] = {'pending': [{'admission_id': 'lost', 'status': 'unknown'}]}
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def rpc(method, **params):
        calls.append(method)
        if method == 'session.resume':
            entered.set()
            await release.wait()
            return {'pending': []}
        assert method == 'prompt.submit'
        await agent._project({'session_id': 's', 'type': 'message.complete',
                              'admission_id': 'next', 'payload': {'outcome': 'completed'}})
        return {'admission_id': 'next'}

    agent._gateway.rpc.side_effect = rpc
    prompt = asyncio.create_task(agent.prompt([acp.text_block('new input')], 's'))
    await asyncio.wait_for(entered.wait(), 5)
    await agent.cancel('s')
    release.set()
    assert (await asyncio.wait_for(prompt, 5)).stop_reason == 'cancelled'
    assert calls == ['session.resume']


@pytest.mark.asyncio
async def test_resumed_unknown_turn_names_the_existing_discard_recovery():
    # ACP has no discard affordance: the resume notice and the refusal must name the exact
    # admission and the classic chat control (``/discard`` -> prompt.resolve_unknown) that clears it.
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._conn = AsyncMock()
    lost = 'admission-lost-0123456789abcdef'
    agent._gateway.rpc.side_effect = lambda method, **params: {
        'session.info': {}, 'config.set': {}, 'session.resume': {
            'session_id': 's', 'messages': [], 'prompts': [],
            'pending': [{'admission_id': lost, 'status': 'unknown', 'execution_generation': 4}]},
    }[method]
    await agent.load_session(cwd='/', session_id='s')
    notice = agent._conn.session_update.await_args.kwargs['update'].content.text
    with pytest.raises(GatewayClientError) as refused:
        await agent.prompt([acp.text_block('new input')], 's')
    for text in (notice, str(refused.value)):
        assert 'sessions discard s' in text and f'/discard {lost}' in text, text
        assert 'do not resend' in text
    assert 'nothing was submitted' in str(refused.value)
    # The prompt refreshed the owner's pending rows; no prompt was submitted.
    asked = [call.args[0] for call in agent._gateway.rpc.await_args_list]
    assert asked == ['session.info', 'session.resume', 'config.set', 'session.resume']


@pytest.mark.asyncio
async def test_resumed_unknown_turn_can_be_discarded_without_leaving_the_editor():
    # The editor needs the same fenced recovery as classic chat. It must use the
    # unknown row's generation, not the session's current generation.
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    lost = 'admission-lost-0123456789abcdef'
    unknown = {'session_id': 's', 'messages': [], 'prompts': [],
               'pending': [{'admission_id': lost, 'status': 'unknown', 'execution_generation': 4}]}
    cleared = {**unknown, 'pending': []}
    agent._snapshots['s'] = unknown
    resolved = []

    def rpc(method, **params):
        if method == 'prompt.resolve_unknown':
            resolved.append(params)
            return {'admission_id': lost, 'status': 'terminal', 'outcome': 'interrupted'}
        return cleared if resolved else unknown
    agent._gateway.rpc.side_effect = rpc

    response = await agent.prompt([acp.text_block(f'/discard {lost}')], 's')

    assert response.stop_reason == 'end_turn'
    assert agent._snapshots['s']['pending'] == []
    # The pre-prompt refresh asks the owner first (another viewer may have resolved it already).
    assert agent._gateway.rpc.await_args_list == [
        call('session.resume', session_id='s'),
        call('session.resume', session_id='s'),
        call('prompt.resolve_unknown', session_id='s', admission_id=lost, execution_generation=4),
        call('session.resume', session_id='s'),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('yield_between', [False, True])
async def test_editor_own_lost_turn_reports_unknown_not_end_turn(yield_between):
    # The prompt's own started turn is lost: whichever frame wakes the prompt first, the
    # editor gets the unknown-execution error, never a successful end_turn.
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._snapshots['s'] = {}

    async def lose_worker():
        await agent._project({'session_id': 's', 'type': 'session.info',
                              'payload': {'pending': [{'admission_id': 'ours', 'status': 'unknown'}]}})
        if yield_between:
            # session.info alone must wake and settle the prompt before the completion arrives.
            await async_wait_until(submitted.done, what='the prompt woke on session.info')
        await agent._project({'session_id': 's', 'type': 'message.complete', 'admission_id': 'ours',
                              'payload': {'outcome': 'unknown', 'text': 'Worker execution is unknown.'}})

    agent._gateway.rpc.return_value = {'admission_id': 'ours'}
    submitted = asyncio.create_task(agent.prompt([acp.text_block('work')], 's'))
    await asyncio.sleep(0)
    await lose_worker()
    with pytest.raises(GatewayClientError, match='unknown_execution'):
        await asyncio.wait_for(submitted, 5)
