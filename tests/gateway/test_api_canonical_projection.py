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

PNG = base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\x00' * 64).decode()


def _turn_runner(owner, ref):
    from gateway.run_turn_runner import TurnRunner
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(_native_slack_task_cards=False, _voice_ack_guild=[None], progress_mode='off')
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
    _, ref, _row = admitted
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
    _, ref, _row = admitted
    launch, _queue = _launch(api, admitted, run_id='run_room')
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


def test_run_status_reports_waiting_for_approval_while_a_prompt_is_pending(api, owner):
    """GET /v1/runs/{id} said `running` while an approval was pending; the documented run state
    (and main's in-memory run store) is `waiting_for_approval`, which UIs key their prompt on."""
    from gateway.platforms.api_server_authority_runs import run_projection
    from hermes_state_runtime import claim_session_input
    _, ref, _ = admit_api_turn(api, session_id='approval-status', request_id='run_wait', user_message='hello',
                               conversation_history=[])
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=ref.session_id)
    assert run_projection(api, 'run_wait')['status'] == 'running'
    live = owner.sessions[ref.session_id]
    owner.register_approval(ref.session_id, row['generation'], live.route, {'request_id': 'approve-me', 'command': 'x'})
    live.controls.remote_responders['approve-me'] = lambda *answer: None
    projected = run_projection(api, 'run_wait')
    assert [p['prompt_id'] for p in projected['pending_controls']] == ['approve-me']
    assert projected['status'] == 'waiting_for_approval'


@pytest.mark.asyncio
@pytest.mark.parametrize('path,body,marker', [
    ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'hi'}], 'stream': True},
     '"reasoning_content": "THINKING"'),
    ('/v1/responses', {'input': 'hi', 'stream': True}, 'response.reasoning_summary_text.delta'),
])
async def test_canonical_sse_projects_the_turn_reasoning(api, owner, path, body, marker):
    """Chat and Responses SSE under the authority get the agent's reasoning, like the direct
    path: the real per-turn wiring hands the admission's observers the reasoning callback."""
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_api_turn import api_execution

    async def handle(event):
        from gateway.session_results import execution_result
        sid = event.source.chat_id
        turn = _turn_runner(owner, SimpleNamespace(session_id=sid))
        agent = SimpleNamespace(reasoning_callback=None)  # AIAgent's own default
        holder = SimpleNamespace(
            _ctx=SimpleNamespace(
                progress_callback=None, native_tool_start_callback=None, voice_ack_callback=None,
                _voice_ack_guild=[None], voice_turn=False, _native_slack_task_cards=False,
                native_tool_complete_callback=None, _step_callback_sync=None,
                _hooks_ref=SimpleNamespace(loaded_hooks=[]), _status_callback_sync=None, _event_callback_sync=None,
                _status_adapter=None, session_key='', user_config={}, source=event.source,
                mute_notification_reply=False, _thinking_enabled=False, agent_holder=[None], tools_holder=[None],
                process_task_id=None, process_baseline=None, run_generation=0),
            _approval_owner=turn._approval_owner, combined_tool_start_callback=None,
            combined_tool_complete_callback=None,
            _runner=SimpleNamespace(_service_tier=None, _consume_pending_turn_sidecar_notes=lambda key: []),
            _make_bg_review_callbacks=lambda: (lambda message: None, lambda: None),
            _merge_turn_request_overrides=TurnRunner._merge_turn_request_overrides,
            _clarify_callback_sync=None, _notice_callback_sync=None,
            _attach_session_title_callback=lambda agent, ctx: None)
        assert api_execution.get() is not None
        TurnRunner._wire_turn_agent_callbacks(holder, agent, {}, None, None, None, False)
        if agent.reasoning_callback:  # AIAgent._fire_reasoning_delta calls it only when wired
            agent.reasoning_callback('THINKING')
        execution_result.get()['result'] = {'final_response': 'ANSWER', 'messages': []}
        return 'ANSWER'
    owner.runner._handle_message = handle
    owner.adopt_agent = lambda *args: None
    app = web.Application()
    app.router.add_post(path, getattr(api, '_handle_chat_completions' if 'chat' in path else '_handle_responses'))
    async with TestClient(TestServer(app)) as client:
        response = await client.post(path, json=body)
        text = await response.text()
    assert response.status == 200, text
    assert marker in text and 'THINKING' in text, text


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['legacy', 'canonical'])
async def test_api_turn_selects_the_requests_registered_browser_controller(api, owner, monkeypatch, path):
    """The extension controller registered for the API session's server-derived principal is the
    one the turn selects, on the legacy executor and under the authority alike."""
    from gateway import browser_control_broker as broker_mod
    from gateway.config import PlatformConfig
    from gateway.run import GatewayRunner
    from gateway.session import SessionContext
    from gateway.session_context import clear_session_vars
    from tools.browser_extension_router import extension_controller_available
    monkeypatch.setattr(broker_mod, 'browser_control_enabled', lambda *a: True)
    adapter = api if path == 'canonical' else _api_server.APIServerAdapter(PlatformConfig(enabled=True))
    adapter._api_key = 'controller-selection-key'
    broker = broker_mod.get_browser_control_broker()
    broker.attach(broker_mod.ControllerScope(
        principal_id=adapter._derive_browser_control_principal('default'), profile_id='default',
        session_id='browser-sel', controller_id='ext-1', transport_family='local-api',
        capabilities=frozenset({'browser_navigate'})), lambda frame: None)
    selected = []
    if path == 'canonical':
        async def handle(event):
            from gateway.session_results import execution_result
            context = SessionContext(source=event.source, connected_platforms=[], home_channels={},
                                     session_key='k', session_id=event.source.chat_id)
            tokens = GatewayRunner._set_session_env(owner.runner, context)
            try:
                selected.append(extension_controller_available('browser_navigate'))
            finally:
                clear_session_vars(tokens)
            execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
            return 'ok'
        owner.runner._handle_message = handle
    else:
        def run_conversation(**kwargs):
            selected.append(extension_controller_available('browser_navigate'))
            return {'final_response': 'ok', 'messages': []}
        monkeypatch.setattr(adapter, '_create_agent', lambda **kw: SimpleNamespace(run_conversation=run_conversation))
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post('/v1/chat/completions', adapter._handle_chat_completions)
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'hi'}]},
                                         headers={'Authorization': 'Bearer controller-selection-key',
                                                  'X-Hermes-Session-Id': 'browser-sel'})
            assert response.status == 200, await response.text()
    finally:
        broker.reset()
    assert selected == [True]
