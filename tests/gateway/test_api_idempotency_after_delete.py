"""An Idempotency-Key stays exactly-once after its session is deleted.

Deleting the session used to drop the admission row ``_one_target`` consults, so the same
key aimed at another session ran the work a second time.
"""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


_API_KEY = 'idempotency-after-delete-key'


def _execute(owner, calls):
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        text = 'answer ' + event.text
        execution_result.get()['result'] = {'final_response': text, 'messages': []}
        return text
    owner.runner._handle_message = handle


def _app(api):
    app = web.Application()
    app.router.add_post('/v1/chat/completions', api._handle_chat_completions)
    app.router.add_delete('/api/sessions/{session_id}', api._handle_delete_session)
    return app


async def _post(client, session_id, *, key='once', content='same body'):
    body = {'model': 'hermes-agent', 'messages': [{'role': 'user', 'content': content}]}
    response = await client.post(
        '/v1/chat/completions', json=body,
        headers={'Authorization': 'Bearer ' + _API_KEY, 'Idempotency-Key': key,
                 'X-Hermes-Session-Id': session_id})
    payload = await response.json()
    return response.status, payload


def _content(payload):
    return payload['choices'][0]['message']['content']


@pytest.mark.asyncio
async def test_deleted_session_idempotency_key_does_not_run_again(api, owner):
    """Same body and key: A runs once, B conflicts, deleting A does not let B run it."""
    calls = []
    api._api_key = _API_KEY
    _execute(owner, calls)
    async with TestClient(TestServer(_app(api))) as client:
        status, first = await _post(client, 'session-a')
        assert status == 200, first
        original = _content(first)

        status, replay = await _post(client, 'session-a')
        assert status == 200, replay
        assert _content(replay) == original

        status, conflict = await _post(client, 'session-b')
        assert status == 409, conflict
        assert conflict['error']['code'] == 'admission_conflict'

        deleted = await client.delete(
            '/api/sessions/session-a', headers={'Authorization': 'Bearer ' + _API_KEY})
        assert deleted.status == 200, await deleted.text()
        assert (await deleted.json())['deleted'] is True

        status, again = await _post(client, 'session-b')
        assert status == 409, again
        assert again['error']['code'] == 'admission_conflict'
        assert calls == ['same body']

        status, same = await _post(client, 'session-a')
        assert status == 200, same
        assert _content(same) == original

        status, fresh = await _post(client, 'session-b', key='fresh-key', content='new body')
        assert status == 200, fresh
        assert _content(fresh) == 'answer new body'
    assert calls == ['same body', 'new body']
