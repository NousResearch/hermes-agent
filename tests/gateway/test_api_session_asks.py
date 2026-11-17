"""Session asks: a turn's pending approval, question and secret prompts as API objects."""
import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server as _api_server
from gateway.platforms.api_server_runs import _RunLaunch, _execute_run
from gateway.session_api_turn import admit_api_turn
from tests.waiting import async_wait_until

HEADERS = {'Authorization': 'Bearer sk-asks'}


def _app(api):
    app = web.Application()
    for method, path, handler in api._http_route_table():
        if '/asks' in path:
            app.router.add_route(method, path, handler)
    return app


def _start(api, owner, session_id, run_id, work):
    """Run *work(generation)* in the agent thread of a real API turn on *session_id*."""
    admitted = admit_api_turn(api, session_id=session_id, request_id=run_id, user_message='hi',
                              conversation_history=[])
    queue = asyncio.Queue()
    api._run_streams[run_id], api._run_statuses[run_id] = queue, {}
    launch = _RunLaunch(api, run_id, queue, admitted[1].session_id, None, False, 'hi', [], False,
                        agent_kwargs={}, request_profile=None, admission=admitted)
    seen = {}

    async def handle(event):
        from gateway.session_results import execution_result
        generation = owner.db.get_session(admitted[1].session_id)['runtime_generation']
        seen['result'] = await asyncio.to_thread(work, generation)
        execution_result.get()['result'] = {'final_response': 'done'}
        return 'done'
    owner.runner._handle_message = handle
    return asyncio.ensure_future(_execute_run(api, launch, _api_server=_api_server)), seen


@pytest.mark.asyncio
async def test_pending_approval_is_listed_and_answered_through_its_native_waiter(api, owner):
    """A dangerous-command approval the turn blocks on is a pending ask; answering it ``once`` over
    the session route reaches the real approval wait, and the ask then reads answered by the API."""
    from tools.approval_gateway_wait import _await_gateway_decision
    api._api_key = 'sk-asks'

    def work(generation):
        return _await_gateway_decision('asks-route', lambda data: owner.register_approval(
            'asks-sess', generation, 'asks-route', data), {'command': 'rm -rf /srv/data', 'description': 'delete'})
    run, seen = _start(api, owner, 'asks-sess', 'run_asks', work)
    async with TestClient(TestServer(_app(api))) as client:
        async def pending():
            listed = await (await client.get('/api/sessions/asks-sess/asks', headers=HEADERS)).json()
            return listed['data']
        await async_wait_until(pending, timeout=10, what='the approval ask listed as pending')
        [ask] = await pending()
        assert ask['kind'] == 'approval' and ask['status'] == 'pending'
        assert ask['payload']['command'] == 'rm -rf /srv/data' and 'once' in ask['payload']['choices']
        assert (await client.get('/api/sessions/asks-sess/asks', headers={})).status == 401
        bad = await client.post(f"/api/sessions/asks-sess/asks/{ask['id']}/answer", json={'choice': 'maybe'},
                                headers=HEADERS)
        assert bad.status == 400
        answered = await client.post(f"/api/sessions/asks-sess/asks/{ask['id']}/answer", json={'choice': 'once'},
                                     headers=HEADERS)
        assert answered.status == 200, await answered.text()
        body = await answered.json()
        assert (body['status'], body['closed_by'], body['answer']) == ('answered', 'api', {'choice': 'once'})
        await asyncio.wait_for(run, timeout=10)
        assert seen['result']['choice'] == 'once' and seen['result']['resolved'] is True
        again = await client.post(f"/api/sessions/asks-sess/asks/{ask['id']}/answer", json={'choice': 'deny'},
                                  headers=HEADERS)
        assert again.status == 409 and (await again.json())['error']['code'] == 'ask_closed'
        assert (await client.get('/api/sessions/asks-sess/asks', headers=HEADERS)).status == 200
        closed = await (await client.get('/api/sessions/asks-sess/asks?status=answered', headers=HEADERS)).json()
        assert [a['id'] for a in closed['data']] == [ask['id']]
        assert (await client.get('/api/sessions/nope/asks', headers=HEADERS)).status == 404


@pytest.mark.asyncio
async def test_question_dismiss_skips_and_a_secret_value_is_never_accepted(api, owner):
    """Dismissing a clarify question sends the clarify skip; a secret ask refuses an answer (the value
    is entered only in an attached client's masked prompt) and a dismiss settles it with no value."""
    from gateway import session_server_requests
    from tools import clarify_gateway
    api._api_key = 'sk-asks'
    transport = type('T', (), {'write': lambda self, frame, overflow=None: True})()

    def work(generation):
        entry = clarify_gateway.register('clarify-asks', 'asks-route', 'Which db?', ['pg', 'sqlite'])
        owner.register_clarify('q-sess', generation, entry)
        reply = clarify_gateway.wait_for_response('clarify-asks', 10)
        events = owner.sessions['q-sess'].event_stream
        events.fanout.attach(transport)
        value = session_server_requests.ask_viewer(events, 'q-sess', 'secret',
                                                   {'env_var': 'TOKEN', 'prompt': 'Paste it'}, timeout=10)
        return reply, value
    run, seen = _start(api, owner, 'q-sess', 'run_q', work)
    async with TestClient(TestServer(_app(api))) as client:
        async def pending(kind):
            listed = await (await client.get('/api/sessions/q-sess/asks', headers=HEADERS)).json()
            return [a for a in listed['data'] if a['kind'] == kind]
        await async_wait_until(lambda: pending('question'), timeout=10, what='the question ask')
        [question] = await pending('question')
        assert question['payload']['choices'] == ['pg', 'sqlite']
        dismissed = await client.post(f"/api/sessions/q-sess/asks/{question['id']}/dismiss", headers=HEADERS)
        assert dismissed.status == 200, await dismissed.text()
        assert (await dismissed.json())['status'] == 'dismissed'
        await async_wait_until(lambda: pending('secret'), timeout=10, what='the secret ask')
        [secret] = await pending('secret')
        assert secret['payload']['env_var'] == 'TOKEN'
        refused = await client.post(f"/api/sessions/q-sess/asks/{secret['id']}/answer", json={'value': 'sk-live'},
                                    headers=HEADERS)
        assert refused.status == 400 and (await refused.json())['error']['code'] == 'answer_with_secret'
        gone = await client.post(f"/api/sessions/q-sess/asks/{secret['id']}/dismiss", headers=HEADERS)
        assert (await gone.json())['status'] == 'dismissed'
        await asyncio.wait_for(run, timeout=10)
        assert seen['result'] == (clarify_gateway.SKIPPED, '')
        everything = await (await client.get('/api/sessions/q-sess/asks?status=all', headers=HEADERS)).json()
        assert 'sk-live' not in str(everything)
        assert {a['status'] for a in everything['data']} == {'dismissed'}
