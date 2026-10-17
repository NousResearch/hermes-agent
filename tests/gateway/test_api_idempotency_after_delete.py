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
    return response.status, await response.json()


async def _delete(client, session_id):
    deleted = await client.delete('/api/sessions/' + session_id, headers={'Authorization': 'Bearer ' + _API_KEY})
    assert deleted.status == 200, await deleted.text()
    assert (await deleted.json())['deleted'] is True


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
        status, conflict = await _post(client, 'session-b')
        assert status == 409 and conflict['error']['code'] == 'admission_conflict', conflict

        await _delete(client, 'session-a')
        status, again = await _post(client, 'session-b')
        assert status == 409, again
        assert again['error']['code'] == 'admission_conflict'
        assert calls == ['same body']

        status, same = await _post(client, 'session-a')
        assert status == 200 and _content(same) == original, same
        assert owner.db.get_session('session-a') is None

        status, fresh = await _post(client, 'session-b', key='fresh-key', content='new body')
        assert status == 200 and _content(fresh) == 'answer new body', fresh
    assert calls == ['same body', 'new body']


@pytest.mark.asyncio
async def test_a_live_session_that_already_ran_the_key_still_replays_it(api, owner):
    """A session that ran the key while the deleted one's tombstone was unread (code from before
    this check) still replays its own completion; only a session that never ran it conflicts."""
    from hermes_state_terminal import ADMISSION_PREFIX
    calls = []
    api._api_key = _API_KEY
    _execute(owner, calls)
    async with TestClient(TestServer(_app(api))) as client:
        status, first = await _post(client, 'session-a')
        assert status == 200, first
        await _delete(client, 'session-a')
        with owner.db._read_ctx() as conn:
            saved = list(conn.execute('SELECT key, value FROM state_meta WHERE key GLOB ?', (ADMISSION_PREFIX + '*',)))
        assert saved
        owner.db._execute_write(lambda conn: conn.execute('DELETE FROM state_meta WHERE key GLOB ?',
                                                          (ADMISSION_PREFIX + '*',)))
        status, live = await _post(client, 'session-b')
        assert status == 200, live
        owner.db._execute_write(lambda conn: conn.executemany(
            'INSERT OR REPLACE INTO state_meta(key, value) VALUES(?, ?)', saved))

        status, again = await _post(client, 'session-b')
        assert status == 200 and _content(again) == _content(live), (status, again)
        status, other = await _post(client, 'session-c')
        assert status == 409 and other['error']['code'] == 'admission_conflict', other
        status, changed = await _post(client, 'session-b', content='other body')
        assert status == 409 and changed['error']['code'] == 'admission_conflict', changed
    assert calls == ['same body', 'same body']
