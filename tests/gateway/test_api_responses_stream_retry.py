"""An exact streaming Responses retry replays the recorded stream, never a new admission."""
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer



def _events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: ')]


@pytest.mark.asyncio
async def test_exact_streaming_retry_on_named_conversation_replays_the_recorded_stream(api, owner):
    calls = []

    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        execution_result.get()['result'] = {'final_response': 'streamed reply', 'messages': []}
        return 'streamed reply'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    body = {'input': 'stream named', 'conversation': 'named', 'stream': True}
    async with TestClient(TestServer(app)) as client:
        streams = []
        for _ in range(2):
            resp = await client.post('/v1/responses', json=body, headers={'Idempotency-Key': 'stream-retry'})
            assert resp.status == 200 and resp.headers['Content-Type'].startswith('text/event-stream')
            streams.append(_events(await resp.text()))
        # Transport mode is not inference identity: a nonstreaming exact retry replays the same envelope.
        plain = await client.post('/v1/responses', json={**body, 'stream': False},
                                  headers={'Idempotency-Key': 'stream-retry'})
        assert plain.status == 200
        plain_body = await plain.json()
    first, retry = streams
    assert [e['type'] for e in first] == [e['type'] for e in retry]
    assert first[-1]['type'] == retry[-1]['type'] == 'response.completed'
    assert retry[-1]['response'] == first[-1]['response'] == plain_body
    assert calls == ['stream named']


@pytest.mark.asyncio
async def test_changed_explicit_history_conflicts_with_same_responses_idempotency_key(api, owner):
    calls = []

    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        execution_result.get()['result'] = {'final_response': 'reply', 'messages': []}
        return 'reply'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    body = {
        'input': 'question',
        'conversation_history': [{'role': 'user', 'content': 'history A'}],
    }
    headers = {'Idempotency-Key': 'history-sensitive-retry'}

    async with TestClient(TestServer(app)) as client:
        first = await client.post('/v1/responses', json=body, headers=headers)
        assert first.status == 200
        changed = await client.post('/v1/responses', json={
            **body,
            'conversation_history': [{'role': 'user', 'content': 'history B'}],
        }, headers=headers)
        payload = await changed.json()

    assert changed.status == 409
    assert payload['error']['code'] == 'admission_conflict'
    assert calls == ['question']


@pytest.mark.asyncio
async def test_a_different_body_under_an_in_flight_key_is_refused_and_the_first_keeps_it(api, owner):
    """One Idempotency-Key names one request. A different body arriving while the first is still
    running (here chained onto another session, so the canonical FIFO cannot see the clash) is
    refused before admission, and the key keeps replaying the first request's response."""
    import asyncio
    calls, release = [], asyncio.Event()

    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        if event.text == 'first':
            await release.wait()
        execution_result.get()['result'] = {'final_response': 'reply to ' + event.text, 'messages': []}
        return 'reply to ' + event.text

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    headers = {'Idempotency-Key': 'one-request'}
    async with TestClient(TestServer(app)) as client:
        seed = await (await client.post('/v1/responses', json={'input': 'seed'})).json()
        first = asyncio.ensure_future(client.post('/v1/responses', json={'input': 'first'}, headers=headers))
        async with asyncio.timeout(10):
            while 'first' not in calls:
                await asyncio.sleep(0.02)
        other = await client.post('/v1/responses', json={'input': 'other', 'previous_response_id': seed['id']},
                                  headers=headers)
        other_body = await other.json()
        release.set()
        first = await first
        first_body = await first.json()
        retry = await client.post('/v1/responses', json={'input': 'first'}, headers=headers)
        retry_body = await retry.json()
    assert other.status == 409 and other_body['error']['code'] == 'admission_conflict'
    assert first.status == 200 and retry.status == 200 and retry_body == first_body
    assert calls == ['seed', 'first']


def test_idempotency_record_keeps_its_first_writer(tmp_path):
    from gateway.platforms.api_server_response_store import ResponseStore
    store = ResponseStore(db_path=str(tmp_path / 'responses.db'))
    try:
        settled = {'fingerprint': 'A', 'response': {'id': 'a'}}
        store.put('idem:k', settled)
        store.put('idem:k', {'fingerprint': 'B', 'response': {'id': 'b'}})
        store.put('idem:k', {'fingerprint': 'A', 'response': {'id': 'late'}})
        assert store.get('idem:k') == settled
    finally:
        store.close()
