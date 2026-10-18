"""An Idempotency-Key retry replays only the same request, including the ``tools`` and ``tool_choice`` it sends.

Chat Completions left both out of its canonical admission identity, and Responses left out
``tool_choice``, so a keyed retry that changed them replayed the first answer. Their identity is
canonical JSON, so the same request with its object members in another order still replays. Keys
stored before they joined the identity keep replaying their exact retries.
"""
import json
import uuid

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

_API_KEY = 'idempotency-tools-key'
_SESSION = 'tools-session'
_TOOL = {'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object', 'properties': {}}}}
# A lone surrogate is a valid JSON escape: the digest of a tool set that carries one still encodes.
_REFUND = {'type': 'function', 'function': {'name': 'refund', 'description': '\ud800',
                                            'parameters': {'type': 'object', 'properties': {}}}}
_CHAT = {'model': 'hermes-agent', 'messages': [{'role': 'user', 'content': 'charge the card'}],
         'tools': [_TOOL], 'tool_choice': 'auto'}
_RESPONSE = {'input': 'charge the card', 'tools': [{'type': 'function', 'name': 'lookup'}],
             'tool_choice': {'type': 'function', 'name': 'lookup'}}
_HEADERS = {'Authorization': 'Bearer ' + _API_KEY, 'Idempotency-Key': 'pay-once', 'X-Hermes-Session-Id': _SESSION}


def _reordered(body):
    """The same JSON with every object's members in another order."""
    reordered = json.loads(json.dumps(body, sort_keys=True))
    assert reordered == body and json.dumps(reordered) != json.dumps(body)
    return reordered


def _reordered_tool_choice(body):
    return {**body, 'tool_choice': _reordered(body['tool_choice'])}


def _app(api, owner, calls):
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        execution_result.get()['result'] = {'final_response': 'charged', 'messages': []}
        return 'charged'
    owner.runner._handle_message = handle
    api._api_key = _API_KEY
    app = web.Application()
    app.router.add_post('/v1/chat/completions', api._handle_chat_completions)
    app.router.add_post('/v1/responses', api._handle_responses)
    app.router.add_delete('/api/sessions/{session_id}', api._handle_delete_session)
    return app


async def _chat(client, body, *, stream=False):
    """-> ``(answer, refusal)``. A streaming refusal is the last chunk of an HTTP 200 stream."""
    response = await client.post('/v1/chat/completions', json={**body, 'stream': stream}, headers=_HEADERS)
    if not stream:
        data = await response.json()
        if response.status != 200:
            return None, (response.status, data['error']['code'])
        return data['choices'][0]['message']['content'], None
    assert response.status == 200
    chunks = [json.loads(line[6:]) for line in (await response.text()).splitlines()
              if line.startswith('data: {')]
    chunks = [chunk for chunk in chunks if 'choices' in chunk]
    if chunks[-1]['choices'][0]['finish_reason'] == 'error':
        return None, (200, chunks[-1]['error']['message'])
    return ''.join(chunk['choices'][0]['delta'].get('content') or '' for chunk in chunks), None


def _store_chat_admission_as_before_the_digest(owner):
    """Rewrite the keyed chat admission to what an older release stored: no tools digest."""
    from hermes_state_runtime import _json, admission_fingerprint

    def write(conn):
        row = conn.execute("SELECT admission_id, target_session_id, payload_json, intent FROM session_admissions "
                           "WHERE request_id LIKE 'chat:%'").fetchone()
        payload = json.loads(row['payload_json'])
        payload.pop('api_tools_v1', None)
        digest = admission_fingerprint(canonical_target=row['target_session_id'],
                                       payload={'input': payload, 'intent': row['intent']})
        conn.execute('UPDATE session_admissions SET payload_json=?, payload_digest=? WHERE admission_id=?',
                     (_json(payload), digest, row['admission_id']))
    owner.db._execute_write(write)


def _store_response_key_as_before_tool_choice(api, body):
    """Rewrite the Responses key record to what an older release stored: no ``tool_choice``."""
    from gateway.platforms.api_server import _make_request_fingerprint
    legacy_keys = ('input', 'instructions', 'previous_response_id', 'conversation', 'conversation_history',
                   'model', 'provider', 'model_options', 'tools')
    fingerprint = _make_request_fingerprint(body, keys=legacy_keys)
    binding = _make_request_fingerprint({'request': fingerprint, 'declared_session_key': None},
                                        keys=('request', 'declared_session_key'))
    store = api._current_response_store()
    key, data = store._conn.execute("SELECT response_id, data FROM responses WHERE response_id LIKE 'idem:%'").fetchone()
    store._conn.execute('UPDATE responses SET data=? WHERE response_id=?',
                        (json.dumps({**json.loads(data), 'fingerprint': fingerprint}), key))
    store._conn.execute('UPDATE response_keys SET fingerprint=? WHERE request_key=?', (binding, key))
    store._conn.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True], ids=['json', 'stream'])
async def test_a_chat_retry_with_other_tools_is_a_conflict_not_a_replay(api, owner, stream):
    calls = []
    async with TestClient(TestServer(_app(api, owner, calls))) as client:
        assert await _chat(client, _CHAT) == ('charged', None)
        refused = (200 if stream else 409, 'admission_conflict')
        for changed in ({'tools': [_TOOL, _REFUND]}, {'tool_choice': 'none'}, {'tools': None, 'tool_choice': None}):
            assert await _chat(client, {**_CHAT, **changed}, stream=stream) == (None, refused), changed
        assert await _chat(client, _reordered(_CHAT), stream=stream) == ('charged', None)
    assert calls == ['charge the card']


@pytest.mark.asyncio
async def test_a_chat_admission_stored_before_the_tools_digest_still_replays_its_exact_retry(api, owner):
    calls = []
    async with TestClient(TestServer(_app(api, owner, calls))) as client:
        assert await _chat(client, _CHAT) == ('charged', None)
        _store_chat_admission_as_before_the_digest(owner)
        assert await _chat(client, _CHAT) == ('charged', None)
        deleted = await client.delete('/api/sessions/' + _SESSION, headers=_HEADERS)
        assert deleted.status == 200, await deleted.text()
        assert await _chat(client, _CHAT) == ('charged', None)
    assert calls == ['charge the card']


@pytest.mark.asyncio
async def test_a_responses_retry_with_another_tool_choice_is_a_conflict_not_a_replay(api, owner):
    calls = []
    async with TestClient(TestServer(_app(api, owner, calls))) as client:
        first = await client.post('/v1/responses', json=_RESPONSE, headers=_HEADERS)
        assert first.status == 200, await first.text()
        for stream in (False, True):
            changed = await client.post('/v1/responses', json={**_RESPONSE, 'tool_choice': 'none', 'stream': stream},
                                        headers=_HEADERS)
            assert changed.status == 409 and (await changed.json())['error']['code'] == 'admission_conflict'
        exact = await client.post('/v1/responses', json=_reordered_tool_choice(_RESPONSE), headers=_HEADERS)
        assert await exact.json() == await first.json()
    assert calls == ['charge the card']


@pytest.mark.asyncio
async def test_a_responses_key_stored_before_tool_choice_joined_still_replays_its_exact_retry(api, owner):
    calls = []
    async with TestClient(TestServer(_app(api, owner, calls))) as client:
        first = await client.post('/v1/responses', json=_RESPONSE, headers=_HEADERS)
        assert first.status == 200, await first.text()
        _store_response_key_as_before_tool_choice(api, _RESPONSE)
        for stream in (False, True):
            exact = await client.post('/v1/responses', json={**_RESPONSE, 'stream': stream}, headers=_HEADERS)
            assert exact.status == 200, await exact.text()
        assert await (await client.post('/v1/responses', json=_RESPONSE, headers=_HEADERS)).json() == await first.json()
    assert calls == ['charge the card']


@pytest.mark.asyncio
async def test_without_a_session_authority_the_responses_cache_keys_on_tool_choice_in_any_member_order(
        tmp_path, monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    calls = []

    async def run_agent(**kwargs):
        calls.append(kwargs['user_message'])
        return {'final_response': 'charged', 'messages': []}, {}
    adapter._run_agent = run_agent
    app = web.Application()
    app.router.add_post('/v1/responses', adapter._handle_responses)
    headers = {'Idempotency-Key': 'cache-' + uuid.uuid4().hex}
    try:
        async with TestClient(TestServer(app)) as client:
            for body in (_RESPONSE, _reordered_tool_choice(_RESPONSE), {**_RESPONSE, 'tool_choice': 'none'}):
                response = await client.post('/v1/responses', json=body, headers=headers)
                assert response.status == 200, await response.text()
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
    assert calls == ['charge the card', 'charge the card']
