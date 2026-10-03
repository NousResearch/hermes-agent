"""An exact streaming Responses retry replays the recorded stream, never a new admission."""
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_cutover_contract import api  # noqa: F401
from tests.gateway.test_api_source_binding import owner  # noqa: F401


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
