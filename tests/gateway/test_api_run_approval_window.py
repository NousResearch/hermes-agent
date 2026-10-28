"""Runs admission -> real blocking native queue -> canonical HTTP controls, without an LLM."""
import asyncio
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server_authority_runs import run_projection
from gateway.session_api_turn import api_execution, api_settings
from gateway.session_contract import SessionRef
from tests.waiting import async_wait_until
from tools.approval_context import _get_approval_timeout
from tools.approval_gateway_wait import _await_gateway_decision


@pytest.mark.asyncio
async def test_scoped_runs_wait_resume_expire_and_retry_without_policy_leak(api, owner):
    outcomes, timeouts = {}, {}

    async def handle(event):
        from gateway.session_results import execution_result
        sid = event.source.chat_id
        generation = owner.db.get_session(sid)['runtime_generation']
        timeouts[sid] = _get_approval_timeout()

        def notify(data):
            owner.register_approval(sid, generation, sid, data)

        result = await asyncio.to_thread(_await_gateway_decision, sid, notify,
            {'command': 'fixture approval only', 'description': 'fixture', 'request_id': sid + '_prompt'})
        outcomes[sid] = result
        execution_result.get()['result'] = {'final_response': 'fixture result', 'completed': True}
        return 'fixture result'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/runs', api._handle_runs)
    app.router.add_get('/v1/runs/{run_id}', api._handle_get_run)
    app.router.add_post('/v1/runs/{run_id}/approval', api._handle_run_approval)
    async with TestClient(TestServer(app)) as client:
        async def create(seconds, key):
            response = await client.post('/v1/runs', json={'input': 'fixture', 'approval_timeout_s': seconds}, headers={'Idempotency-Key': key})
            assert response.status == 202, await response.text()
            return (await response.json())['run_id']

        long = await create(1800, 'long')
        short = await create(1, 'short')
        assert await create(1800, 'long') == long
        for run in (long, short):
            await async_wait_until(lambda: run_projection(api, run)['pending_controls'], what='real native waiter registered')
        pending = await (await client.get('/v1/runs/' + long)).json()
        prompt = pending['pending_controls'][0]
        assert pending['status'] == 'waiting_for_approval'
        assert prompt['timeout_s'] == 1800
        assert 1790 < prompt['expires_at'] - time.time() <= 1800
        assert timeouts[long] == 1800 and timeouts[short] == 1
        # Scoped admission data is durable, but never a session default for internal follow-ups.
        assert 'approval_timeout_s' not in api_settings(owner, SessionRef('default', long))
        assert api_execution.get() is None
        assert _get_approval_timeout() not in (1, 1800)
        response = await client.post('/v1/runs', json={'input': 'fixture', 'approval_timeout_s': 1}, headers={'Idempotency-Key': 'long'})
        assert response.status == 409
        body = {'request_id': prompt['prompt_id'], 'execution_generation': pending['execution_generation'], 'choice': 'once'}
        stale = await client.post('/v1/runs/' + long + '/approval', json={**body, 'execution_generation': -1})
        assert stale.status == 409
        # The one-second native waiter expires naturally while the other run stays alive.
        await async_wait_until(lambda: short in outcomes, timeout=10, what='native timeout')
        assert outcomes[short]['resolved'] is False and outcomes[short]['choice'] is None
        assert run_projection(api, long)['status'] == 'waiting_for_approval'
        assert run_projection(api, long)['pending_controls'][0]['expires_at'] == prompt['expires_at']
        approved = await client.post('/v1/runs/' + long + '/approval', json=body)
        assert approved.status == 200, await approved.text()
        assert (await approved.json())['status'] == 'resolved'
        await async_wait_until(lambda: long in outcomes, what='same native waiter resumed')
        assert outcomes[long]['choice'] == 'once'
        replay = await client.post('/v1/runs/' + long + '/approval', json=body)
        assert replay.status == 409
        for run in (long, short):
            await async_wait_until(lambda: run_projection(api, run)['status'] == 'completed', what='native run terminal')
        late = await client.post('/v1/runs/' + short + '/approval', json={'request_id': short + '_prompt', 'execution_generation': 1, 'choice': 'once'})
        assert late.status == 409


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [None, True, 0, -1, 1801, '1800', 1.5])
async def test_invalid_window_never_admits(api, owner, value):
    app = web.Application()
    app.router.add_post('/v1/runs', api._handle_runs)
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/v1/runs', json={'input': 'fixture', 'approval_timeout_s': value})
        assert response.status == 400
    assert not owner.sessions
