"""Observed peer approvals survive polling failures and refuse stale canonical controls."""
import asyncio
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from gateway.session_api_turn import admit_api_turn
from gateway.session_authority import admission_row
from gateway.session_controls import AuthorityConnection
from gateway.session_hosted_service import CanonicalHostedRoomService
from persistence.gateway.hosted_rooms import HostedTaskIdentity
from tests.tui_gateway.test_hosted_room_two_gateway_peer import _target_app
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


@pytest_asyncio.fixture
async def lender(api, owner, tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'profile_id', str(tmp_path))
    api._api_key = 'target-peer-key-1234567890'
    wire = SimpleNamespace(fail_get=False, calls=[], errors=[], now=0)

    @web.middleware
    async def record(request, handler):
        wire.calls.append((request.method, request.path))
        if wire.fail_get and request.method == 'GET':
            raise web.HTTPServiceUnavailable()
        response = await handler(request)
        if response.status >= 400:
            wire.errors.append(json.loads(response.text)['error']['code'])
        return response

    app = _target_app(api)
    app.middlewares.append(record)
    async with TestServer(app) as server:
        client = PeerRunsHTTPClient(base_url=str(server.make_url('')).rstrip('/'), api_key=api._api_key,
                                   poll_min_seconds=60, poll_max_seconds=60, clock=lambda: wire.now)
        invited = await asyncio.to_thread(client.issue_invitation, room_id='room-1',
            home_install_id='home', authority_gateway_id='home', authority_epoch=1,
            member_id='member-peer', grant_id='grant')
        grant = invited['grant']
        _, ref, _ = admit_api_turn(api, session_id='approval-session', request_id='run_approval',
                                  user_message='run', conversation_history=[])
        row = admission_row(owner.persistence.admissions.claim(epoch=owner.epoch, session_id=ref.session_id))
        scope = SimpleNamespace(headers={'Authorization': 'HermesRoom ' + grant},
                                path='/v1/runs/run_approval', method='GET')
        api._run_owners['run_approval'] = api._run_idempotency_scope(scope)
        live = owner.sessions[ref.session_id]
        owner.register_approval(ref.session_id, row['generation'], live.route, {
            'request_id': 'prompt', 'command': 'echo approved', 'description': 'Test command'})
        answers = []
        live.controls.remote_responders['prompt'] = lambda *answer: answers.append(answer)
        client._runs[('task', 7)] = {
            'run_id': 'run_approval', 'session_id': ref.session_id, 'room_id': 'room-1',
            'target_profile': 'default', 'task_id': 'task', 'execution_generation': 7}
        yield SimpleNamespace(api=api, owner=owner, client=client, wire=wire, row=row, ref=ref,
                              live=live, answers=answers, grant=grant)


async def _observe(lender):
    return await asyncio.to_thread(lender.client.status, room_id='room-1', profile='default',
        session_id=lender.ref.session_id, task_id='task', execution_generation=7, grant=lender.grant)


@pytest.mark.asyncio
@pytest.mark.parametrize('poll', ['failed', 'expired', 'unobserved'])
async def test_approval_posts_observed_identity_without_polling(lender, poll):
    if poll != 'unobserved':
        observed = await _observe(lender)
        assert observed['execution_generation'] == 7
        assert observed['run_execution_generation'] == lender.row['generation']
        assert observed['approval'] == {
            'request_id': 'prompt', 'command': 'echo approved', 'description': 'Test command',
            'choices': ['once', 'deny', 'session', 'always']}
        lender.wire.now = 61
        if poll == 'failed':
            lender.wire.fail_get = True
            with pytest.raises(PeerRunsHTTPError) as failed:
                await _observe(lender)
            assert failed.value.status_code == 503
            lender.wire.fail_get = False
    lender.wire.calls.clear()
    answered = await asyncio.to_thread(lender.client.approve_receipt, task_id='task',
        execution_generation=7, request_id='prompt', choice='once', grant=lender.grant)
    assert answered['status'] == 'resolved'
    assert lender.answers == [('approval', 'prompt', 'once')]
    expected = [('GET', '/v1/runs/run_approval')] if poll == 'unobserved' else []
    assert lender.wire.calls == expected + [('POST', '/v1/runs/run_approval/approval')]


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['stale_generation', 'approval_not_pending', 'approval_not_active'])
async def test_groups_approve_refusal_matches_local_nonpending_approval(lender, monkeypatch, reason):
    observed = await _observe(lender)
    owner, live, row = lender.owner, lender.live, lender.row
    service = owner.hosted_room_service = CanonicalHostedRoomService(owner, asyncio.get_running_loop())
    monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True, 'stopping': False})
    connection = AuthorityConnection(owner, object(), {'user_id': 'alice'})
    service.insert_room(room_id='room-1', name='room-1', members=[], owner=connection.actor.subject)
    params = dict(room_id='room-1', member_id='member-peer', task_id='task',
                  execution_generation=7, request_id='prompt', choice='once')
    request = {'id': 1, 'method': 'groups.approve', 'params': params}
    key = ('room-1', 'member-peer')
    service.peer_routes[key] = SimpleNamespace(grant=lender.grant)
    service.peer_clients[key] = lender.client
    service._set_pending_action(*key, {**params, 'kind': 'approval', 'approval': observed['approval']})
    # The home still runs that attempt (#148877 refuses an approval for an attempt it no longer runs):
    # the refusal under test is the lender's.
    from tui_gateway import hosted_room_service
    attempt = {'identity': HostedTaskIdentity('room-1', 'task', 'thread', 'turn'),
               'execution_generation': 7, 'status': 'running', 'payload': {'target_member_id': 'member-peer'}}
    monkeypatch.setattr(hosted_room_service.driver, 'list_tasks', lambda *_a, **_kw: [attempt])
    if reason == 'stale_generation':
        owner.persistence.admissions.settle(epoch=owner.epoch, admission_id=row['admission_id'],
                                            generation=row['generation'], outcome='interrupted')
    elif reason == 'approval_not_pending':
        live.controls.respond(lender.ref.session_id, row['generation'], 'prompt', {'choice': 'deny'})
        owner.register_approval(lender.ref.session_id, row['generation'], live.route, {
            'request_id': 'replacement', 'command': 'echo replacement'})
        live.controls.remote_responders['replacement'] = lambda *answer: lender.answers.append(answer)
    else:
        # The older handler has no active approval session after its run ends.
        lender.api.gateway_runner = None
        lender.api._set_run_status('run_approval', 'completed')
    lender.wire.now = 61
    await _observe(lender)  # The displayed identity survives a newer status without that prompt.
    before = list(lender.answers)
    lender.wire.calls.clear()
    refused = await connection.dispatch(request)
    assert refused['error']['code'] == 4001
    assert refused['error']['message'] == 'room approval is no longer pending'
    assert lender.wire.errors == [reason]
    assert lender.wire.calls == [('POST', '/v1/runs/run_approval/approval')]
    assert lender.answers == before
    if reason == 'approval_not_pending':
        assert live.controls.snapshot(lender.ref.session_id, row['generation'])[0]['prompt_id'] == 'replacement'
    service._set_pending_action(*key, None)
    service.peer_routes.clear()
    service.peer_clients.clear()
    local = await connection.dispatch(request)
    assert refused['error'] == local['error']
