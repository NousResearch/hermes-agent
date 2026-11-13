"""A Responses identity settles from its admission's terminal outcome, even when the streaming
client left before the replay record could be written."""
import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_responses_identity_retention import COMPACT, _compact_rows


def _app(api, owner, calls, release, hold=None):
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        if event.text.startswith('gone'):
            await release.wait()
        if event.text == 'held':
            await hold.wait()
        execution_result.get()['result'] = {'final_response': 'answer ' + event.text, 'messages': []}
        return 'answer ' + event.text
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    app.router.add_delete('/v1/responses/{response_id}', api._handle_delete_response)
    return app


async def _disconnect_mid_stream(client, store, key):
    """Open a stream, read ``response.created``, leave; return once the server saw the loss."""
    response = await client.post('/v1/responses', json={'input': key, 'stream': True},
                                 headers={'Idempotency-Key': key})
    assert response.status == 200
    while True:
        line = (await response.content.readline()).decode()
        if line.startswith('data: '):
            response_id = json.loads(line[6:])['response']['id']
            break
    response.close()
    for _ in range(200):
        stored = store.get(response_id)
        if stored and stored['response']['status'] == 'incomplete':
            return response_id
        await asyncio.sleep(0.05)
    raise AssertionError('the server never observed the disconnect')


async def _await_terminal(owner, count):
    for _ in range(200):
        rows = owner.db._read_all("SELECT status FROM session_admissions WHERE principal_id='api'", ())
        if len(rows) >= count and all(row['status'] == 'terminal' for row in rows):
            return
        await asyncio.sleep(0.05)
    raise AssertionError('the admitted turns never settled')


@pytest.fixture
def fast_keepalive(monkeypatch):
    # An idle stream learns of a closed client from its keepalive write.
    from gateway.platforms import api_server
    monkeypatch.setattr(api_server, 'CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS', 0.0)


@pytest.mark.asyncio
async def test_disconnected_stream_identity_joins_the_bound_and_never_reruns(api, owner, fast_keepalive):
    calls, release = [], asyncio.Event()
    store = api._current_response_store()
    store._max_size, store._max_identities = 1, 2
    async with TestClient(TestServer(_app(api, owner, calls, release))) as client:
        for index in range(4):
            await _disconnect_mid_stream(client, store, f'gone-{index}')
        release.set()
        await _await_terminal(owner, 4)
        for index in range(3):
            response = await client.post('/v1/responses', json={'input': f'after {index}'},
                                         headers={'Idempotency-Key': f'after-{index}'})
            assert response.status == 200
        keys = sorted(row[0].rsplit(':', 1)[1] for row in store._conn.execute('SELECT request_key FROM response_keys'))
        # Finished-but-unobserved identities settle and age out; the bound holds.
        assert keys == ['after-1', 'after-2'], keys
        assert _compact_rows(store)[:2] == [2, 2]
        expired = await client.post('/v1/responses', json={'input': 'gone-0', 'stream': True},
                                    headers={'Idempotency-Key': 'gone-0'})
        assert expired.status == 409
        assert (await expired.json())['error']['code'] == 'admission_conflict'
    assert calls == [f'gone-{index}' for index in range(4)] + [f'after {index}' for index in range(3)]


@pytest.mark.asyncio
async def test_delete_retires_a_disconnected_stream_identity_but_not_a_running_one(api, owner, fast_keepalive):
    calls, release, hold = [], asyncio.Event(), asyncio.Event()
    store = api._current_response_store()
    store._max_size = 1
    release.set()
    async with TestClient(TestServer(_app(api, owner, calls, release, hold))) as client:
        done = await _disconnect_mid_stream(client, store, 'gone-done')
        await _await_terminal(owner, 1)
        busy = await _disconnect_mid_stream(client, store, 'held')
        # The client deletes the response it walked away from: its identity goes everywhere.
        assert (await client.delete('/v1/responses/' + done)).status == 200
        assert not [row for row in store._conn.execute(
            "SELECT request_key FROM response_keys WHERE request_key LIKE '%gone-done'")]
        # A deleted key is still never permission to run again.
        retry = await client.post('/v1/responses', json={'input': 'gone-done', 'stream': True},
                                  headers={'Idempotency-Key': 'gone-done'})
        assert retry.status == 409
        # Running work keeps its identity: delete drops the snapshot, never the running key.
        assert (await client.delete('/v1/responses/' + busy)).status == 200
        keys = [row[0].rsplit(':', 1)[1] for row in store._conn.execute('SELECT request_key FROM response_keys')]
        assert keys == ['held'], keys
        hold.set()
        await _await_terminal(owner, 2)
    assert calls == ['gone-done', 'held']
