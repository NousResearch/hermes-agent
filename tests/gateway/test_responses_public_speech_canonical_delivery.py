"""Canonical public projection characterization with real compression and concurrent GET."""
import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server_public_speech import PublicSpeechStream
from tests.gateway.test_api_server import _create_app
from tests.gateway.test_responses_compressed_chain import _compressing_app
from tests.gateway.test_responses_public_speech import _events


@pytest.mark.asyncio
async def test_public_compression_terminal_identity_replay_and_next_post_use_one_logical_owner(api, owner):
    calls = []
    app = _compressing_app(api, owner, calls)
    app.router.add_get('/v1/responses/{response_id}', api._handle_get_response)
    async with TestClient(TestServer(app)) as client:
        body = {'input': 'one', 'stream': True, 'public_commentary': True}
        response = await client.post('/v1/responses', json=body, headers={'Idempotency-Key': 'compressed-public'})
        wire = await response.read()
        final = _events(wire)[-1][1]['response']
        original = response.headers['X-Hermes-Session-Id']
        child = original + '-child'
        assert final['hermes']['session_id'] == child
        snapshot = api._current_response_store().get(final['id'])
        assert snapshot['session_id'] == child and snapshot['response'] == final
        retry = await client.post('/v1/responses', json=body, headers={'Idempotency-Key': 'compressed-public'})
        assert await retry.read() == wire
        following = await client.post('/v1/responses', json={**body, 'input': 'two',
            'previous_response_id': final['id']}, headers={'Idempotency-Key': 'compressed-follow'})
        assert following.status == 200 and following.headers['X-Hermes-Session-Id'] == original
        # Canonical admissions remain on the lineage root; the public response still
        # carries the physical effective tip when the runtime reports it.
        assert _events(await following.read())[-1][0] == 'response.completed'
    roots = {row['target_session_id'] for row in owner.db._read_all(
        "SELECT target_session_id FROM session_admissions WHERE principal_id='api'", ())}
    assert roots == {original} and calls == ['one', 'two']


@pytest.mark.asyncio
async def test_canonical_public_get_holds_final_until_terminal_write_and_late_store_failure_is_not_replayed(api, owner, monkeypatch, caplog):
    from tests.gateway.test_responses_public_speech_authority import _producer, PUBLIC, HEADERS
    calls, pending, release = [], asyncio.Event(), asyncio.Event()
    _producer(owner, calls)
    original_write = web.StreamResponse.write

    async def hold_terminal(response, data):
        if b'event: response.completed\n' in data:
            pending.set()
            await release.wait()
        return await original_write(response, data)

    monkeypatch.setattr(web.StreamResponse, 'write', hold_terminal)
    store, put = api._current_response_store(), api._current_response_store().put

    def late_failure(response_id, data):
        if data.get('response', {}).get('status') == 'completed':
            raise RuntimeError('synthetic late snapshot failure')
        return put(response_id, data)

    monkeypatch.setattr(store, 'put', late_failure)
    async with TestClient(TestServer(_create_app(api))) as client:
        response = await client.post('/v1/responses', json=PUBLIC, headers=HEADERS)
        try:
            await asyncio.wait_for(pending.wait(), 5)
            from gateway.platforms.api_server_response_identity import durable_response_id
            from types import SimpleNamespace
            scope = api._run_idempotency_scope(SimpleNamespace(headers=HEADERS))
            response_id = durable_response_id(f'idem:{scope}:speech-turn')
            snapshot = await (await client.get('/v1/responses/' + response_id)).json()
            assert snapshot['status'] == 'in_progress'
            assert all(item.get('phase') == 'commentary' for item in snapshot['output'])
            assert 'Authoritative answer.' not in json.dumps(snapshot)
        finally:
            release.set()
        wire = await response.read()
        assert [name for name, _ in _events(wire) if name in
                ('response.completed', 'response.failed', 'response.incomplete')] == ['response.completed']
        assert store.get(response_id)['response']['status'] == 'in_progress'
        assert 'snapshot persistence failed' in caplog.text
    assert calls == ['question']
