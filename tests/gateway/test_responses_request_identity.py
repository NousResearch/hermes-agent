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
    from gateway.platforms.api_server_idempotency import make_request_fingerprint
    from gateway.platforms.api_server_openai_routes import _responses_fingerprint_keys
    legacy_keys = ("input", "instructions", "previous_response_id", "conversation", "conversation_history",
                   "model", "provider", "model_options", "tools")
    body = {"input": "hi", "model": "m"}
    assert make_request_fingerprint(body, keys=_responses_fingerprint_keys(body)) == \
        make_request_fingerprint(body, keys=legacy_keys)
    truncated = dict(body, truncation="auto")
    assert make_request_fingerprint(truncated, keys=_responses_fingerprint_keys(truncated)) != \
        make_request_fingerprint(body, keys=_responses_fingerprint_keys(body))


@pytest.mark.asyncio
@pytest.mark.parametrize(("first_fields", "retry_fields"), [
    ({"conversation_history": [{"role": "user", "content": "secret-A"}]},
     {"conversation_history": [{"role": "user", "content": "secret-B"}]}),
    ({}, {"truncation": "auto"}),
    ({}, {"metadata": {"source": "room-a"}}),
    ({}, {"store": False}),
])
async def test_durable_response_retry_rejects_changed_context(api, owner, first_fields, retry_fields):
    api._api_key = "identity-fixture"
    calls = []

    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        execution_result.get()["result"] = {"final_response": "answer", "messages": []}
        return "answer"

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post("/v1/responses", api._handle_responses)
    headers = {"Authorization": "Bearer identity-fixture", "Idempotency-Key": "same-context"}
    async with TestClient(TestServer(app)) as client:
        first = await client.post("/v1/responses", json={"input": "work", **first_fields}, headers=headers)
        retry = await client.post("/v1/responses", json={"input": "work", **retry_fields}, headers=headers)
        retry_data = await retry.json()

    assert first.status == 200
    assert retry.status == 409
    assert retry_data["error"]["code"] == "admission_conflict"
    assert calls == ["work"]


@pytest.mark.asyncio
async def test_durable_response_retry_rejects_changed_route_credentials(api, owner):
    api._api_key = "identity-fixture"
    api._direct_model_requests = True
    route = {"model": "provider-model", "provider": "openai", "api_key": "route-secret-a",
             "base_url": "https://a.example/v1"}
    api._model_routes = {"alias": route}
    calls = []

    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        execution_result.get()["result"] = {"final_response": "answer", "messages": []}
        return "answer"

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post("/v1/responses", api._handle_responses)
    headers = {"Authorization": "Bearer identity-fixture", "Idempotency-Key": "same-route"}
    async with TestClient(TestServer(app)) as client:
        first = await client.post("/v1/responses", json={"model": "alias", "input": "work"}, headers=headers)
        first_data = await first.json()
        api._model_routes["alias"] = {
            **route, "api_key": "route-secret-b", "base_url": "https://b.example/v1"}
        retry = await client.post("/v1/responses", json={"model": "alias", "input": "work"}, headers=headers)
        retry_data = await retry.json()

    assert first.status == 200, first_data
    assert retry.status == 409
    assert retry_data["error"]["code"] == "admission_conflict"
    assert calls == ["work"]


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
