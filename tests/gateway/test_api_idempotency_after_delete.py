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


def _strip_api_request_keys(db):
    """A store deleted before the api-request pointer existed has only the terminal admission."""
    from hermes_state_terminal import API_REQUEST_BACKFILL_KEY, API_REQUEST_PREFIX
    db._execute_write(lambda conn: conn.execute(
        'DELETE FROM state_meta WHERE key GLOB ? OR key=?',
        (API_REQUEST_PREFIX + '*', API_REQUEST_BACKFILL_KEY)))


@pytest.mark.asyncio
async def test_preexisting_terminal_tombstone_blocks_another_session(api, owner):
    """A delete from before the api-request key still consumes the Idempotency-Key.

    Same-session replay uses the older identity tombstone. Another session must get
    409, not a second execution, because the terminal-admission row is still there.
    """
    from gateway.platforms.api_server_response_admissions import _request_admission_rows
    from hermes_state_terminal import ADMISSION_PREFIX
    calls = []
    api._api_key = _API_KEY
    _execute(owner, calls)
    async with TestClient(TestServer(_app(api))) as client:
        status, first = await _post(client, 'session-a')
        assert status == 200, first
        original = _content(first)
        with owner.db._read_ctx() as conn:
            request_id = conn.execute(
                "SELECT request_id FROM session_admissions WHERE target_session_id='session-a'").fetchone()[0]
        deleted = await client.delete(
            '/api/sessions/session-a', headers={'Authorization': 'Bearer ' + _API_KEY})
        assert deleted.status == 200, await deleted.text()
        _strip_api_request_keys(owner.db)
        with owner.db._read_ctx() as conn:
            assert conn.execute(
                "SELECT 1 FROM state_meta WHERE key GLOB ?", (ADMISSION_PREFIX + '*',)).fetchone()
        assert _request_admission_rows(owner.db, request_id), request_id

        status, same = await _post(client, 'session-a')
        assert status == 200, same
        assert _content(same) == original
        assert owner.db.get_session('session-a') is None
        assert calls == ['same body']

        status, again = await _post(client, 'session-b')
        assert status == 409, again
        assert again['error']['code'] == 'admission_conflict'
        from hermes_state_terminal import api_request_key
        with owner.db._read_ctx() as conn:
            assert conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                                (api_request_key(request_id),)).fetchone()
    assert calls == ['same body']


def _saved_terminal_rows(db):
    from hermes_state_terminal import ADMISSION_PREFIX
    with db._read_ctx() as conn:
        return list(conn.execute(
            'SELECT key, value FROM state_meta WHERE key GLOB ?', (ADMISSION_PREFIX + '*',)))


def _hide_terminal_admissions(db):
    """Drop the tombstone and pointer so a second session can be admitted, as pre-upgrade code did."""
    from hermes_state_terminal import ADMISSION_PREFIX, API_REQUEST_BACKFILL_KEY, API_REQUEST_PREFIX
    saved = _saved_terminal_rows(db)
    db._execute_write(lambda conn: conn.execute(
        'DELETE FROM state_meta WHERE key GLOB ? OR key GLOB ? OR key=?',
        (ADMISSION_PREFIX + '*', API_REQUEST_PREFIX + '*', API_REQUEST_BACKFILL_KEY)))
    return saved


def _restore_terminal_rows(db, saved):
    db._execute_write(lambda conn: conn.executemany(
        'INSERT OR REPLACE INTO state_meta(key, value) VALUES(?, ?)', saved))


@pytest.mark.asyncio
async def test_preupgrade_live_session_replays_its_own_completion(api, owner):
    """A second session that already ran the key before the upgrade still replays its completion.

    The deleted session's terminal admission must not turn that retry into 409. Another session
    that never ran the key still conflicts, and the handler does not run again.
    """
    calls = []
    api._api_key = _API_KEY
    _execute(owner, calls)
    async with TestClient(TestServer(_app(api))) as client:
        status, first = await _post(client, 'session-a')
        assert status == 200, first
        original = _content(first)
        deleted = await client.delete(
            '/api/sessions/session-a', headers={'Authorization': 'Bearer ' + _API_KEY})
        assert deleted.status == 200, await deleted.text()
        saved = _hide_terminal_admissions(owner.db)

        status, live = await _post(client, 'session-b')
        assert status == 200, live
        live_answer = _content(live)
        with owner.db._read_ctx() as conn:
            live_id = conn.execute(
                "SELECT admission_id FROM session_admissions WHERE target_session_id='session-b'"
            ).fetchone()[0]
        _restore_terminal_rows(owner.db, saved)
        _strip_api_request_keys(owner.db)

        status, same = await _post(client, 'session-a')
        assert status == 200 and _content(same) == original, same
        assert owner.db.get_session('session-a') is None

        status, other = await _post(client, 'session-c')
        assert status == 409 and other['error']['code'] == 'admission_conflict', other

        status, again = await _post(client, 'session-b')
        assert status == 200 and _content(again) == live_answer, (status, again)
        with owner.db._read_ctx() as conn:
            row = conn.execute(
                "SELECT admission_id, status FROM session_admissions WHERE target_session_id='session-b'"
            ).fetchone()
        assert row[0] == live_id and row[1] == 'terminal'
        status, changed = await _post(client, 'session-b', content='other body')
        assert status == 409 and changed['error']['code'] == 'admission_conflict', changed
    assert calls == ['same body', 'same body']
