"""An in-flight Responses identity cannot acquire a second target via a moving conversation."""
import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_pending_retry_pins_named_conversation_target(api, owner):
    entered, duplicate, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        if event.text == 'slow':
            entered.set()
            if calls.count('slow') > 1:
                duplicate.set()
            await release.wait()
        execution_result.get()['result'] = {'final_response': 'reply', 'messages': []}
        return 'reply'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    first_body = {'input': 'slow', 'conversation': 'shared'}
    first_headers = {'Idempotency-Key': 'pending-original'}
    requests = []
    async with TestClient(TestServer(app)) as client:
        try:
            requests.append(asyncio.create_task(client.post('/v1/responses', json=first_body, headers=first_headers)))
            await asyncio.wait_for(entered.wait(), 10)
            advanced = await client.post('/v1/responses', json={'input': 'fast', 'conversation': 'shared'},
                                         headers={'Idempotency-Key': 'concurrent-other'})
            assert advanced.status == 200
            requests.append(asyncio.create_task(client.post('/v1/responses', json=first_body, headers=first_headers)))
            try:
                await asyncio.wait_for(duplicate.wait(), 2)
            except TimeoutError:
                pass
            release.set()
            responses = await asyncio.gather(*requests)
            assert all(response.status == 200 for response in responses)
            assert calls.count('slow') == 1, 'retry executed the same request on the newly mapped session'
            assert await responses[0].json() == await responses[1].json()
            assert responses[0].headers['X-Hermes-Session-Id'] == responses[1].headers['X-Hermes-Session-Id']
        finally:
            release.set()
            await asyncio.gather(*requests, return_exceptions=True)


@pytest.mark.asyncio
async def test_unknown_retry_recovers_admission_after_pointer_commit_loss(api, owner):
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import begin_runtime_epoch, recover_session_inputs
    from gateway.platforms.api_server_response_store import ResponseStore
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def handle(event):
        calls.append(event.text)
        started.set()
        await release.wait()
        return 'done'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    body, headers = {'input': 'uncertain', 'conversation': 'shared'}, {'Idempotency-Key': 'lost-owner'}
    async with TestClient(TestServer(app)) as client:
        first = asyncio.create_task(client.post('/v1/responses', json=body, headers=headers))
        try:
            await asyncio.wait_for(started.wait(), 10)
            row = owner.db._read_one('SELECT * FROM session_admissions')
            task = owner.sessions[row['target_session_id']].task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            owner.epoch = begin_runtime_epoch(owner.db, instance_id='restarted')
            recover_session_inputs(owner.db, epoch=owner.epoch)
            owner._pause(SessionRef(owner.profile_id, row['target_session_id']), 'unknown_execution')
            assert (await first).status == 409
            # Crash after the ledger commit but before its response-store pointer was saved.
            store = api._current_response_store()
            store._conn.execute('DELETE FROM response_admissions')
            store._conn.commit()
            path = store._db_path
            store.close()
            api._response_store = ResponseStore(db_path=path)
            api._response_store.put('different', {'session_id': 'different-target', 'conversation_history': []})
            api._response_store.set_conversation('shared', 'different')
            retry = await client.post('/v1/responses', json=body, headers=headers)
            assert retry.status == 409
            assert (await retry.json())['error']['code'] == 'unknown_execution'
            assert calls == ['uncertain']
            assert owner.db._read_one('SELECT COUNT(*) FROM session_admissions')[0] == 1
            assert api._response_store._conn.execute('SELECT admission_id FROM response_admissions').fetchone()[0] == row['admission_id']
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_validation_refusal_does_not_reserve_retry_key(api, owner):
    calls = []
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': []}
        return 'answer'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    headers = {'Idempotency-Key': 'corrected-request'}
    async with TestClient(TestServer(app)) as client:
        refused = await client.post('/v1/responses', json={'input': 42}, headers=headers)
        assert refused.status == 400
        assert api._current_response_store()._conn.execute('SELECT COUNT(*) FROM response_keys').fetchone()[0] == 0
        accepted = await client.post('/v1/responses', json={'input': 'valid'}, headers=headers)
        assert accepted.status == 200
    assert calls == ['valid']


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_concurrent_exact_responses_share_terminal_envelope(api, owner, stream):
    import json
    started, release = asyncio.Event(), asyncio.Event()
    async def handle(event):
        from gateway.session_results import execution_result
        started.set()
        await release.wait()
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': [
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"a"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'read'},
            {'role': 'assistant', 'content': 'answer'}]}
        return 'answer'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    body, headers = {'input': 'work', 'stream': stream}, {'Idempotency-Key': 'concurrent-exact'}
    pending = []
    async with TestClient(TestServer(app)) as client:
        try:
            pending.append(asyncio.create_task(client.post('/v1/responses', json=body, headers=headers)))
            await asyncio.wait_for(started.wait(), 10)
            pending.append(asyncio.create_task(client.post('/v1/responses', json=body, headers=headers)))
            # Observe both HTTP requests before the turn can settle.
            deadline = asyncio.get_running_loop().time() + 5
            while sum(len(observers) for observers in getattr(owner, 'api_observers', {}).values()) < 2:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(.01)
            release.set()
            responses = await asyncio.gather(*pending)
            if stream:
                frames = [[json.loads(line[6:]) for line in (await response.text()).splitlines()
                           if line.startswith('data: ')] for response in responses]
                values = [events[-1]['response'] for events in frames]
                assert frames[0][0]['response'] == frames[1][0]['response']
            else:
                values = [await response.json() for response in responses]
            assert values[0] == values[1]
            retry = await client.post('/v1/responses', json={**body, 'stream': False}, headers=headers)
            assert await retry.json() == values[0]
            assert owner.db._read_one('SELECT COUNT(*) FROM session_admissions')[0] == 1
        finally:
            release.set()
            await asyncio.gather(*pending, return_exceptions=True)
