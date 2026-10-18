"""Accepted Idempotency-Key identity has a bounded lifecycle beside the response-body LRU."""
import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

COMPACT = ('response_keys', 'response_admissions', 'response_output_order')


def _compact_rows(store):
    return [store._conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in COMPACT]


def _app(api, owner, calls, gate=None):
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        if gate is not None and event.text == 'slow':
            gate[0].set()
            await gate[1].wait()
        execution_result.get()['result'] = {'final_response': 'answer ' + event.text, 'messages': []}
        return 'answer ' + event.text
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    app.router.add_delete('/v1/responses/{response_id}', api._handle_delete_response)
    return app


@pytest.mark.asyncio
async def test_unique_keys_and_deletion_keep_compact_identity_bounded(api, owner):
    calls = []
    store = api._current_response_store()
    store._max_size, store._max_identities = 1, 3
    async with TestClient(TestServer(_app(api, owner, calls))) as client:
        originals = []
        for index in range(6):
            response = await client.post('/v1/responses', json={'input': f'turn {index}'},
                                         headers={'Idempotency-Key': f'unique-{index}'})
            assert response.status == 200
            originals.append(await response.json())
        # Steady unique traffic: one body, and the compact identity stays at its documented bound.
        assert len(store) == 1
        assert _compact_rows(store) == [3, 3, 3]
        # Inside the window an exact retry still replays after its body was evicted.
        retry = await client.post('/v1/responses', json={'input': 'turn 5'},
                                  headers={'Idempotency-Key': 'unique-5'})
        assert retry.status == 200 and await retry.json() == originals[5]
        # An expired key is a clear conflict, never a second execution.
        expired = await client.post('/v1/responses', json={'input': 'turn 0'},
                                    headers={'Idempotency-Key': 'unique-0'})
        assert expired.status == 409
        assert (await expired.json())['error']['code'] == 'admission_conflict'
        # Public deletion retires the identity in every compact table.
        for value in originals[3:]:
            assert (await client.delete('/v1/responses/' + value['id'])).status == 200
        assert _compact_rows(store) == [0, 0, 0]
        deleted = await client.post('/v1/responses', json={'input': 'turn 5'},
                                    headers={'Idempotency-Key': 'unique-5'})
        assert deleted.status == 409
    assert calls == [f'turn {index}' for index in range(6)]


@pytest.mark.asyncio
async def test_pending_identity_survives_identity_pressure(api, owner):
    calls, gate = [], (asyncio.Event(), asyncio.Event())
    store = api._current_response_store()
    store._max_size, store._max_identities = 1, 1
    slow_headers = {'Idempotency-Key': 'pending-slow'}
    async with TestClient(TestServer(_app(api, owner, calls, gate))) as client:
        slow = asyncio.create_task(client.post('/v1/responses', json={'input': 'slow'}, headers=slow_headers))
        try:
            await asyncio.wait_for(gate[0].wait(), 10)
            for index in range(3):
                response = await client.post('/v1/responses', json={'input': f'fast {index}'},
                                             headers={'Idempotency-Key': f'fast-{index}'})
                assert response.status == 200
            # Settled identities aged out around it; the running admission's identity did not.
            keys = sorted(row[0].rsplit(':', 1)[1] for row in store._conn.execute('SELECT request_key FROM response_keys'))
            assert keys == ['fast-2', 'pending-slow']
        finally:
            gate[1].set()
        original = await (await slow).json()
        retry = await client.post('/v1/responses', json={'input': 'slow'}, headers=slow_headers)
        assert retry.status == 200 and await retry.json() == original
    assert calls == ['slow', 'fast 0', 'fast 1', 'fast 2']
