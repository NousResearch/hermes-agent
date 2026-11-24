"""Responses persistence runs off the event loop and stays readable at once (#142641).

``ResponseStore.put`` serialises the whole chained conversation_history and commits it; on the
gateway's loop that stalled every other request (the Desktop WebSocket included) for as long as
the write took. Moving it to a worker must not loosen ordering: the id a request returns is
readable by the very next request.
"""
import asyncio
import json
import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

_API_KEY = 'test-key'
_AUTH = {'Authorization': 'Bearer ' + _API_KEY}


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_response_store_writes_leave_the_loop_and_read_back_at_once(api, owner, monkeypatch, stream):
    from gateway.platforms.api_server_response_store import ResponseStore
    from gateway.session_results import execution_result

    async def handle(event):
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    owner.runner._handle_message = handle
    api._api_key = _API_KEY
    loop_thread = threading.get_ident()
    put_threads = []
    real_put = ResponseStore.put

    def _put(self, response_id, data):
        put_threads.append(threading.get_ident())
        return real_put(self, response_id, data)

    monkeypatch.setattr(ResponseStore, 'put', _put)
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    app.router.add_get('/v1/responses/{response_id}', api._handle_get_response)
    async with TestClient(TestServer(app)) as client:
        for n, key in enumerate(('first', 'second')):
            body = {'input': f'turn {n}', 'conversation': 'project', 'stream': stream}
            posted = await client.post('/v1/responses', json=body, headers={**_AUTH, 'Idempotency-Key': key})
            text = await posted.text()
            assert posted.status == 200, text
            if stream:
                done = [json.loads(line[6:]) for line in text.splitlines()
                        if line.startswith('data: ') and '"response.completed"' in line]
                response_id = done[-1]['response']['id']
            else:
                response_id = json.loads(text)['id']
            fetched = await client.get('/v1/responses/' + response_id, headers=_AUTH)
            assert fetched.status == 200, f'response {response_id} not readable right after it was returned'
            assert (await fetched.json())['status'] == 'completed'
    assert put_threads, 'nothing was persisted'
    assert loop_thread not in put_threads, 'ResponseStore.put ran on the event loop thread'


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_store_false_keeps_no_snapshot_and_no_conversation_pointer(api, owner, stream):
    from gateway.session_results import execution_result

    async def handle(event):
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    owner.runner._handle_message = handle
    api._api_key = _API_KEY
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    app.router.add_get('/v1/responses/{response_id}', api._handle_get_response)
    async with TestClient(TestServer(app)) as client:
        body = {'input': 'private question', 'conversation': 'ephemeral', 'stream': stream, 'store': False}
        posted = await client.post('/v1/responses', json=body, headers=_AUTH)
        text = await posted.text()
        assert posted.status == 200, text
        if stream:
            done = [json.loads(line[6:]) for line in text.splitlines()
                    if line.startswith('data: ') and '"response.completed"' in line]
            response_id = done[-1]['response']['id']
        else:
            response_id = json.loads(text)['id']
        fetched = await client.get('/v1/responses/' + response_id, headers=_AUTH)
        assert fetched.status == 404, 'store:false response was retained'
    assert api._response_store.get_conversation('ephemeral') is None


@pytest.mark.asyncio
async def test_keyed_admission_and_retry_never_take_the_store_lock_on_the_loop(api, owner):
    """The store lock is held by workers for a whole persistence unit; a loop-side wait on it
    freezes every other session for as long as that write takes."""
    from gateway.session_results import execution_result

    async def handle(event):
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    owner.runner._handle_message = handle
    api._api_key = _API_KEY
    store = api._current_response_store()
    lock_threads = []

    class _SpyLock:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            lock_threads.append(threading.get_ident())
            return self._inner.__enter__()

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    store._lock = _SpyLock(store._lock)
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    async with TestClient(TestServer(app)) as client:
        for _ in range(2):  # a fresh keyed admission, then its exact retry
            posted = await client.post('/v1/responses', json={'input': 'hi'},
                                       headers={**_AUTH, 'Idempotency-Key': 'lock-probe'})
            assert posted.status == 200, await posted.text()
    assert lock_threads, 'the store was never used'
    assert threading.get_ident() not in lock_threads, 'the event loop waited on the ResponseStore lock'


@pytest.mark.asyncio
async def test_disconnect_never_waits_on_a_worker_held_store_lock_on_the_loop(api):
    """A persistence worker keeps running after its handler is cancelled; adapter teardown that
    took the store lock on the loop froze every profile's loop (and its own cleanup timeout)."""
    store = api._current_response_store()
    held, release, released = threading.Event(), threading.Event(), threading.Event()

    def _hold_store():
        held.set()
        release.wait(3)  # safety deadline: on the bug the loop is stuck until this expires
        released.set()

    worker = asyncio.create_task(store.off_loop(_hold_store))
    assert await asyncio.to_thread(held.wait, 5)
    teardown = asyncio.create_task(api.disconnect())
    loop_ran_while_held = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(lambda: loop_ran_while_held.set_result(not released.is_set()))
    try:
        assert await loop_ran_while_held, 'disconnect waited on the worker-held ResponseStore lock on the event loop'
    finally:
        release.set()
        await worker
        await teardown
    with pytest.raises(Exception):
        store._conn.execute('SELECT 1')  # the connection was still closed
