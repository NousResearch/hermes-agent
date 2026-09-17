"""Canonical API admissions project the real turn: media, tool payloads, controls, outcomes."""
import asyncio
import base64
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server as _api_server
from gateway.platforms.api_server_runs import _RunLaunch, _execute_run
from gateway.session_api_turn import admit_api_turn
from gateway.session_ingress import execute_admission
from tests.gateway.test_api_cutover_contract import api  # noqa: F401
from tests.gateway.test_api_source_binding import owner  # noqa: F401

PNG = base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\x00' * 64).decode()


def _turn_runner(owner, ref):
    from gateway.run_turn_runner import TurnRunner
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(_native_slack_task_cards=False, _voice_ack_guild=[None])
    turn._approval_owner = (owner, ref.session_id, owner.db.get_session(ref.session_id)['runtime_generation'])
    return turn


def _launch(api, admitted, run_id='run_test'):
    queue = asyncio.Queue()
    api._run_streams[run_id] = queue
    api._run_statuses[run_id] = {}
    launch = _RunLaunch(api, run_id, queue, admitted[1].session_id, None, False, 'hello', [], False,
                        agent_kwargs={}, request_profile=None, browser_control_principal=None,
                        browser_control_transport_family=None, admission=admitted)
    return launch, queue


def _events(queue):
    events = []
    while not queue.empty():
        item = queue.get_nowait()
        if item is not None:
            events.append(item)
    return events


@pytest.mark.asyncio
async def test_api_images_are_committed_media_in_canonical_history(api, owner):
    content = [{'type': 'text', 'text': 'what is this'},
               {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + PNG}}]
    args = dict(session_id='vision', request_id='img-1', user_message=content, conversation_history=[])
    _, ref, row = admit_api_turn(api, **args)
    media = row['payload']['api_turn_v1']['media']
    assert len(media) == 1 and media[0]['size'] == 72
    # An exact retry commits the same bytes to the same reference: no admission conflict.
    assert admit_api_turn(api, **args)[2]['admission_id'] == row['admission_id']
    seen = {}

    async def handle(event):
        from gateway.session_api_turn import api_execution
        from gateway.session_results import execution_result
        seen['text'] = event.text
        seen['content'] = api_execution.get()['content']
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'
    owner.runner._handle_message = handle
    await execute_admission(owner, ref, row)
    committed = media[0]['path']
    assert f'[Image attached at: {committed}]' in seen['text']
    assert seen['content'][0]['text'].endswith(f'[Image attached at: {committed}]')
    assert seen['content'][1]['image_url']['url'].startswith('data:image/png;base64,')


@pytest.mark.asyncio
async def test_initialization_failure_is_a_failed_receipt_not_completed(api, owner):
    _, ref, row = admit_api_turn(api, session_id='init', user_message='hello', conversation_history=[])

    async def handle(event):
        return 'Sorry, I encountered an unexpected error.'  # handler apology, no TurnRunner result
    owner.runner._handle_message = handle
    await execute_admission(owner, ref, row)
    result = owner.pending_results[row['admission_id']]['result']
    assert result['failed'] is True and result['completed'] is False
    assert result['final_response'] == '' and 'unexpected error' in result['error']


@pytest.mark.asyncio
async def test_run_sse_carries_controls_tool_payloads_and_authoritative_cancellation(api, owner):
    from tools import clarify_gateway
    admitted = admit_api_turn(api, session_id='sse', user_message='hello', conversation_history=[])
    _, ref, row = admitted
    launch, queue = _launch(api, admitted)
    tool_payloads = []

    async def handle(event):
        from gateway.session_results import execution_result
        turn = _turn_runner(owner, ref)
        entry = clarify_gateway.register('clarify-1', 'api', 'Which one?', ['a', 'b'])
        owner.register_clarify(ref.session_id, turn._approval_owner[2], entry)
        turn.combined_tool_start_callback('call-1', 'read_file', {'path': '/tmp/x'})
        turn.combined_tool_complete_callback('call-1', 'read_file', {'path': '/tmp/x'}, {'content': 'body'})
        entry.event.set()
        # Interrupted over another transport (WS /stop): this adapter never marked the run stopping.
        execution_result.get()['result'] = {'final_response': 'partial', 'interrupted': True}
        return 'partial'
    owner.runner._handle_message = handle
    api._make_run_event_callback = lambda run_id, loop: _wrap(api, run_id, loop, tool_payloads)
    await _execute_run(api, launch, _api_server=_api_server)
    await asyncio.sleep(0)  # call_soon_threadsafe projections land on the loop
    events = _events(queue)
    names = [e['event'] for e in events]
    clarify = next((e for e in events if e['event'] == 'clarify.request'), None)
    assert clarify is not None, events
    assert clarify['prompt_id'] == 'clarify-1' and clarify['choices'] == ['a', 'b']
    assert [e['tool_call_id'] for e in events if e['event'].startswith('tool.')] == ['call-1', 'call-1']
    assert tool_payloads == [('read_file', {'path': '/tmp/x'}), ('read_file', {'path': '/tmp/x'})]
    assert 'run_test' not in api._stopping_run_ids
    assert 'run.cancelled' in names and 'run.completed' not in names


def _wrap(api, run_id, loop, sink):
    from gateway.platforms.api_server_runs import _make_run_event_callback
    inner = _make_run_event_callback(api, run_id, loop, _api_server=_api_server)

    def callback(event_type, tool_name=None, preview=None, args=None, **kwargs):
        sink.append((tool_name, args))
        return inner(event_type, tool_name, preview, args, **kwargs)
    return callback


@pytest.mark.asyncio
async def test_responses_streaming_projects_real_tool_arguments_and_results(api, owner):
    from gateway.session_api_turn import observe_api_turn
    admitted = admit_api_turn(api, session_id='stream', user_message='hello', conversation_history=[])
    _, ref, _ = admitted
    seen = []

    async def handle(event):
        from gateway.session_results import execution_result
        turn = _turn_runner(owner, ref)
        turn.combined_tool_complete_callback('call-9', 'read_file', {'path': 'a.txt'}, 'file body')
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'
    owner.runner._handle_message = handle
    await observe_api_turn(admitted, tool_complete_callback=lambda *a: seen.append(a))
    assert seen == [('call-9', 'read_file', {'path': 'a.txt'}, 'file body')]


@pytest.mark.asyncio
async def test_exact_responses_retry_replays_before_conversation_expansion(api, owner):
    calls = []

    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        execution_result.get()['result'] = {'final_response': 'reply', 'messages': []}
        return 'reply'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    async with TestClient(TestServer(app)) as client:
        statuses = []
        for _ in range(2):
            resp = await client.post('/v1/responses', json={'input': 'hello', 'conversation': 'named'},
                                     headers={'Idempotency-Key': 'retry-1'})
            statuses.append((resp.status, (await resp.json()).get('status')))
    assert statuses == [(200, 'completed'), (200, 'completed')]
    assert calls == ['hello']


@pytest.mark.asyncio
async def test_cancelled_observer_is_unregistered_and_never_breaks_the_survivor(api, owner):
    """Two request tasks observe one admission. Cancelling one (client disconnect) must drop
    exactly its observer entry; the other keeps streaming, and a raising callback on the
    owner path is isolated from canonical execution."""
    from gateway.session_api_turn import observe_api_turn
    admitted = admit_api_turn(api, session_id='observers', user_message='hello', conversation_history=[])
    _, ref, row = admitted
    started, release = asyncio.Event(), asyncio.Event()
    survivor, cancelled_saw = [], []

    def broken(*args):
        raise RuntimeError('client sink is gone')

    async def handle(event):
        from gateway.session_results import execution_result
        started.set()
        await release.wait()
        turn = _turn_runner(owner, ref)
        turn.combined_tool_complete_callback('call-1', 'read_file', {'path': 'a.txt'}, 'body')
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'
    owner.runner._handle_message = handle
    first = asyncio.ensure_future(observe_api_turn(admitted, tool_complete_callback=lambda *a: cancelled_saw.append(a)))
    second = asyncio.ensure_future(observe_api_turn(admitted, tool_complete_callback=lambda *a: survivor.append(a)))
    await asyncio.wait_for(started.wait(), timeout=5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert len(owner.api_observers[row['admission_id']]) == 1
    # A sink that raises on the owner path must not abort the turn or starve the survivor.
    third = asyncio.ensure_future(observe_api_turn(admitted, tool_complete_callback=broken))
    await asyncio.sleep(0)
    release.set()
    (result, _), (third_result, _) = await asyncio.wait_for(asyncio.gather(second, third), timeout=5)
    assert result['final_response'] == third_result['final_response'] == 'ok'
    assert survivor == [('call-1', 'read_file', {'path': 'a.txt'}, 'body')]
    assert cancelled_saw == []
    assert row['admission_id'] not in owner.api_observers


@pytest.mark.asyncio
async def test_room_grant_answers_the_clarify_prompt_its_run_raised(api, owner, tmp_path):
    """A room-scoped run's clarify prompt is answerable by the same room grant that dispatched
    it (no gateway API key), and a revoked grant can no longer answer."""
    import time
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
    from gateway.platforms.api_server_authority_runs import run_projection
    from tools import clarify_gateway
    api._api_key = 'sk-secret'
    now = time.time()
    grant = issue_room_grant(
        api._room_grant_secret(), grant_id='grant-clarify', room_id='room-1', home_install_id='install-home',
        authority_gateway_id='install-home', authority_epoch=1, member_id='member-peer',
        target_install_id=hosted_rooms.local_authority_gateway_id(), target_profile='default',
        issued_at=now, ttl_seconds=300, status_expires_at=now + 1000)
    claims = decode_room_grant(api._room_grant_secret(), grant, permission='status')
    hosted_rooms.reserve_peer_room(hosted_rooms.default_db_path(), claims=claims, expires_at=now + 1000)
    headers = {'Authorization': f'HermesRoom {grant}'}
    admitted = admit_api_turn(api, session_id='room-clarify', request_id='run_room', user_message='hello',
                              conversation_history=[])
    _, ref, row = admitted
    launch, queue = _launch(api, admitted, run_id='run_room')
    scope_request = SimpleNamespace(headers=headers, path='/v1/runs/run_room/clarify', method='POST')
    api._run_owners['run_room'] = api._run_idempotency_scope(scope_request)

    async def handle(event):
        from gateway.session_results import execution_result
        turn = _turn_runner(owner, ref)
        entry = clarify_gateway.register('clarify-room', 'api', 'Which one?', ['a', 'b'])
        owner.register_clarify(ref.session_id, turn._approval_owner[2], entry)
        await asyncio.to_thread(entry.event.wait, 10)
        execution_result.get()['result'] = {'final_response': entry.response or 'unanswered'}
        return entry.response
    owner.runner._handle_message = handle
    run = asyncio.ensure_future(_execute_run(api, launch, _api_server=_api_server))
    async with asyncio.timeout(10):
        while not (run_projection(api, 'run_room') or {}).get('pending_controls'):
            await asyncio.sleep(0.02)
    prompt = run_projection(api, 'run_room')['pending_controls'][0]
    body = {'request_id': prompt['prompt_id'], 'execution_generation': prompt['execution_generation'], 'answer': 'b'}
    app = web.Application()
    app.router.add_post('/v1/runs/{run_id}/clarify', api._handle_run_clarify)
    async with TestClient(TestServer(app)) as client:
        answered = await client.post('/v1/runs/run_room/clarify', json=body, headers=headers)
        assert answered.status == 200, await answered.text()
        assert (await answered.json())['status'] == 'resolved'
        await asyncio.wait_for(run, timeout=10)
        assert api._run_statuses['run_room']['output'] == 'b'
        hosted_rooms.revoke_room_grant_scope(hosted_rooms.default_db_path(), claims=claims, expires_at=now + 1000)
        denied = await client.post('/v1/runs/run_room/clarify', json=body, headers=headers)
        assert denied.status == 403
        assert (await denied.json())['error']['code'] == 'room_reauthorization_required'
