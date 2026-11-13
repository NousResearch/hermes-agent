"""No-header Chat Completions requests get independent sessions; explicit retries keep identity."""
import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


_API_KEY = 'implicit-session-after-delete-key'
_OPENING = [{'role': 'system', 'content': 'be brief'}, {'role': 'user', 'content': 'hi'}]


def _execute(owner, calls):
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        text = f'answer {len(calls)}'
        execution_result.get()['result'] = {'final_response': text, 'messages': []}
        return text
    owner.runner._handle_message = handle


def _app(api):
    app = web.Application()
    app.router.add_post('/v1/chat/completions', api._handle_chat_completions)
    app.router.add_delete('/api/sessions/{session_id}', api._handle_delete_session)
    return app


async def _post(client, messages, *, session_id=None, key=None):
    headers = {'Authorization': 'Bearer ' + _API_KEY}
    if session_id:
        headers['X-Hermes-Session-Id'] = session_id
    if key:
        headers['Idempotency-Key'] = key
    response = await client.post('/v1/chat/completions', json={'model': 'hermes-agent', 'messages': messages},
                                 headers=headers)
    return response.status, await response.json(), response.headers.get('X-Hermes-Session-Id')


async def _delete(client, session_id):
    deleted = await client.delete('/api/sessions/' + session_id, headers={'Authorization': 'Bearer ' + _API_KEY})
    assert deleted.status == 200, await deleted.text()
    assert (await deleted.json())['deleted'] is True


def _prune_past_horizon(owner):
    from hermes_state_mutation_retirement import prune_retired_tombstones
    owner.db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))


def _content(payload):
    return payload['choices'][0]['message']['content']


@pytest.fixture
def served(api, owner):
    calls = []
    api._api_key = _API_KEY
    _execute(owner, calls)
    return calls


@pytest.mark.asyncio
async def test_no_header_requests_with_the_same_opening_get_independent_sessions(api, owner, served):
    async with TestClient(TestServer(_app(api))) as client:
        status, first, deleted_sid = await _post(client, _OPENING)
        assert status == 200, first
        await _delete(client, deleted_sid)

        status, opened, new_sid = await _post(client, _OPENING)
        assert status == 200, opened
        assert new_sid and new_sid != deleted_sid
        follow_up = [*_OPENING, {'role': 'assistant', 'content': _content(opened)}, {'role': 'user', 'content': 'next'}]
        status, second, follow_up_sid = await _post(client, follow_up)
        assert status == 200, second
        assert follow_up_sid not in (deleted_sid, new_sid)
        assert served == ['hi', 'hi', 'next']
        assert owner.db.get_session(deleted_sid) is None, 'the deleted session was recreated'
        assert owner.db.get_session(new_sid) is not None


@pytest.mark.asyncio
async def test_an_explicit_session_id_of_a_deleted_session_is_still_refused(api, owner, served):
    async with TestClient(TestServer(_app(api))) as client:
        status, first, deleted_sid = await _post(client, _OPENING)
        assert status == 200, first
        await _delete(client, deleted_sid)
        _prune_past_horizon(owner)
        status, refused, _sid = await _post(client, _OPENING, session_id=deleted_sid)
        assert status != 200 and refused['error']['code'] == 'not_found', refused
    assert served == ['hi']
    assert owner.db.get_session(deleted_sid) is None


@pytest.mark.asyncio
async def test_an_idempotency_key_retry_keeps_its_deleted_session(api, owner, served):
    """An exact retry replays its original answer on the deleted id; a changed payload under the
    same key conflicts. Neither is moved to a new generation or run again."""
    async with TestClient(TestServer(_app(api))) as client:
        status, first, deleted_sid = await _post(client, _OPENING, key='once')
        assert status == 200, first
        await _delete(client, deleted_sid)
        status, replay, replay_sid = await _post(client, _OPENING, key='once')
        assert status == 200 and _content(replay) == _content(first), replay
        assert replay_sid == deleted_sid
        changed_body = [*_OPENING, {'role': 'assistant', 'content': 'x'}, {'role': 'user', 'content': 'changed'}]
        status, changed, _sid = await _post(client, changed_body, key='once')
        assert status == 409 and changed['error']['code'] == 'admission_conflict', changed
    assert served == ['hi']
    assert owner.db.get_session(deleted_sid) is None


@pytest.mark.asyncio
async def test_concurrent_keyed_retries_share_their_session(api, owner, monkeypatch):
    api._api_key = _API_KEY
    started = asyncio.Event()
    retry_resolving = asyncio.Event()
    release = asyncio.Event()
    calls = []
    resolver_calls = 0
    original_resolver = api._implicit_chat_session

    async def observe_resolver(request, session_id):
        nonlocal resolver_calls
        resolver_calls += 1
        if resolver_calls == 2:
            retry_resolving.set()
        return await original_resolver(request, session_id)

    monkeypatch.setattr(api, "_implicit_chat_session", observe_resolver)

    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        started.set()
        await release.wait()
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': []}
        return 'answer'

    owner.runner._handle_message = handle
    headers = {'Authorization': 'Bearer ' + _API_KEY, 'Idempotency-Key': 'concurrent'}
    async with TestClient(TestServer(_app(api))) as client:
        first_task = asyncio.create_task(client.post(
            '/v1/chat/completions', json={'model': 'hermes-agent', 'messages': _OPENING}, headers=headers))
        await asyncio.wait_for(started.wait(), 5)
        second_task = asyncio.create_task(client.post(
            '/v1/chat/completions', json={'model': 'hermes-agent', 'messages': _OPENING}, headers=headers))
        await asyncio.wait_for(retry_resolving.wait(), 5)
        release.set()
        first, second = await asyncio.gather(first_task, second_task)

    assert first.status == second.status == 200
    assert first.headers['X-Hermes-Session-Id'] == second.headers['X-Hermes-Session-Id']
    assert calls == ['hi']


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True], ids=['json', 'stream'])
async def test_a_store_that_cannot_answer_the_lookup_returns_an_openai_error(api, owner, served, monkeypatch, stream):
    """A busy or unreadable store is retryable: 503 ``storage_unavailable`` with Retry-After, the
    same verdict ``store_error_response`` gives every other admission door, never the raw error."""
    import sqlite3

    def unreadable(*_a, **_k):
        raise sqlite3.OperationalError('disk I/O error')

    monkeypatch.setattr('gateway.session_api.implicit_api_session', unreadable)
    async with TestClient(TestServer(_app(api))) as client:
        response = await client.post('/v1/chat/completions',
                                     json={'model': 'hermes-agent', 'messages': _OPENING, 'stream': stream},
                                     headers={'Authorization': 'Bearer ' + _API_KEY})
        assert response.content_type == 'application/json', await response.text()
        payload = await response.json()
        assert response.status == 503 and payload['error']['code'] == 'storage_unavailable', payload
        assert response.headers.get('Retry-After') == '1'
        assert 'disk I/O error' not in json.dumps(payload)
    assert served == []


@pytest.mark.asyncio
async def test_idempotency_key_is_scoped_to_its_session_header(api, owner, served):
    async with TestClient(TestServer(_app(api))) as client:
        status, first, deleted_sid = await _post(client, _OPENING)
        assert status == 200, first
        status, other, _sid = await _post(client, [{'role': 'user', 'content': 'elsewhere'}],
                                          session_id='other-session', key='spent')
        assert status == 200, other
        await _delete(client, deleted_sid)
        status, new_session, _sid = await _post(client, _OPENING, key='spent')
        assert status == 200, new_session
    assert served == ['hi', 'elsewhere', 'hi']
