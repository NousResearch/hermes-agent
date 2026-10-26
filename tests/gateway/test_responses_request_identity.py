"""Canonical response retry identity is fixed before concurrent admission."""
import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_concurrent_changed_conversation_cannot_replace_response_key(api, owner):
    api._api_key = 'identity-fixture'
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        started.set()
        if event.text == 'first':
            await release.wait()
        execution_result.get()['result'] = {'final_response': 'first answer', 'messages': []}
        return 'first answer'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    headers = {'Idempotency-Key': 'same-request', 'X-Hermes-Session-Key': 'first-room',
               'Authorization': 'Bearer identity-fixture'}
    async with TestClient(TestServer(app)) as client:
        first = asyncio.create_task(client.post('/v1/responses', json={'input': 'first'}, headers=headers))
        await asyncio.wait_for(started.wait(), 5)
        try:
            changed = await client.post('/v1/responses', json={'input': 'different'},
                headers={**headers, 'X-Hermes-Session-Key': 'second-room'})
            assert changed.status == 409
        finally:
            release.set()
        original = await first
        assert original.status == 200
        saved = await original.json()
        replay = await client.post('/v1/responses', json={'input': 'first'}, headers=headers)
        assert await replay.json() == saved
        changed_target = await client.post('/v1/responses', json={'input': 'first'},
            headers={**headers, 'X-Hermes-Session-Key': 'second-room'})
        assert changed_target.status == 409
    assert calls == ['first']


def test_unsent_truncation_keeps_the_fingerprint_of_records_stored_before_it_was_keyed():
    from gateway.platforms.api_server import _make_request_fingerprint
    from gateway.platforms.api_server_openai_routes import _responses_fingerprint_keys
    legacy_keys = ("input", "instructions", "previous_response_id", "conversation", "conversation_history",
                   "model", "provider", "model_options", "tools")
    body = {"input": "hi", "model": "m"}
    assert _make_request_fingerprint(body, keys=_responses_fingerprint_keys(body)) == \
        _make_request_fingerprint(body, keys=legacy_keys)
    truncated = dict(body, truncation="auto")
    assert _make_request_fingerprint(truncated, keys=_responses_fingerprint_keys(truncated)) != \
        _make_request_fingerprint(body, keys=_responses_fingerprint_keys(body))


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_claim_recorded_without_its_answer_refuses_an_exact_retry_without_running_it(api, owner, stream):
    import json
    calls = []
    async def handle(event):
        calls.append(event.text)
        return 'answer'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    body, headers = {'input': 'work'}, {'Idempotency-Key': 'claimed-before-upgrade'}
    async with TestClient(TestServer(app)) as client:
        assert (await client.post('/v1/responses', json=body, headers=headers)).status == 200
        store = api._current_response_store()
        key, data = store._conn.execute("SELECT response_id, data FROM responses WHERE response_id LIKE 'idem:%'").fetchone()
        # An older release left only its claim: the outcome of that attempt is unknown.
        store._conn.execute('UPDATE responses SET data=? WHERE response_id=?',
                            (json.dumps({'fingerprint': json.loads(data)['fingerprint']}), key))
        store._conn.commit()
        retry = await client.post('/v1/responses', json={**body, 'stream': stream}, headers=headers)
        assert retry.status == 409
        assert (await retry.json())['error']['code'] == 'unknown_execution'
    assert calls == ['work']
