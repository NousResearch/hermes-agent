"""A Responses identity settles from its admission's terminal outcome, even when the streaming
client left before the replay record could be written."""
import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_responses_identity_retention import COMPACT, _compact_rows


def _app(api, owner, calls, release, hold=None):
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        if event.text.startswith('gone'):
            await release.wait()
        if event.text == 'held':
            await hold.wait()
        execution_result.get()['result'] = {'final_response': 'answer ' + event.text, 'messages': []}
        return 'answer ' + event.text
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    app.router.add_delete('/v1/responses/{response_id}', api._handle_delete_response)
    return app


async def _disconnect_mid_stream(client, store, key):
    """Open a stream, read ``response.created``, leave; return once the server saw the loss."""
    response = await client.post('/v1/responses', json={'input': key, 'stream': True},
                                 headers={'Idempotency-Key': key})
    assert response.status == 200
    while True:
        line = (await response.content.readline()).decode()
        if line.startswith('data: '):
            response_id = json.loads(line[6:])['response']['id']
            break
    response.close()
    for _ in range(200):
        stored = store.get(response_id)
        if stored and stored['response']['status'] == 'incomplete':
            return response_id
        await asyncio.sleep(0.05)
    raise AssertionError('the server never observed the disconnect')


async def _await_terminal(owner, count):
    for _ in range(200):
        rows = owner.db._read_all("SELECT status FROM session_admissions WHERE principal_id='api'", ())
        if len(rows) >= count and all(row['status'] == 'terminal' for row in rows):
            return
        await asyncio.sleep(0.05)
    raise AssertionError('the admitted turns never settled')


@pytest.fixture
def fast_keepalive(monkeypatch):
    # An idle stream learns of a closed client from its keepalive write.
    from gateway.platforms import api_server
    monkeypatch.setattr(api_server, 'CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS', 0.0)


@pytest.mark.asyncio
async def test_disconnected_stream_identity_joins_the_bound_and_never_reruns(api, owner, fast_keepalive):
    calls, release = [], asyncio.Event()
    store = api._current_response_store()
    store._max_size, store._max_identities = 1, 2
    async with TestClient(TestServer(_app(api, owner, calls, release))) as client:
        for index in range(4):
            await _disconnect_mid_stream(client, store, f'gone-{index}')
        release.set()
        await _await_terminal(owner, 4)
        for index in range(3):
            response = await client.post('/v1/responses', json={'input': f'after {index}'},
                                         headers={'Idempotency-Key': f'after-{index}'})
            assert response.status == 200
        keys = sorted(row[0].rsplit(':', 1)[1] for row in store._conn.execute('SELECT request_key FROM response_keys'))
        # Finished-but-unobserved identities settle and age out; the bound holds.
        assert keys == ['after-1', 'after-2'], keys
        assert _compact_rows(store)[:2] == [2, 2]
        expired = await client.post('/v1/responses', json={'input': 'gone-0', 'stream': True},
                                    headers={'Idempotency-Key': 'gone-0'})
        assert expired.status == 409
        assert (await expired.json())['error']['code'] == 'admission_conflict'
    assert calls == [f'gone-{index}' for index in range(4)] + [f'after {index}' for index in range(3)]


@pytest.mark.asyncio
async def test_delete_retires_a_disconnected_stream_identity_but_not_a_running_one(api, owner, fast_keepalive):
    calls, release, hold = [], asyncio.Event(), asyncio.Event()
    store = api._current_response_store()
    store._max_size = 1
    release.set()
    async with TestClient(TestServer(_app(api, owner, calls, release, hold))) as client:
        done = await _disconnect_mid_stream(client, store, 'gone-done')
        await _await_terminal(owner, 1)
        busy = await _disconnect_mid_stream(client, store, 'held')
        # The client deletes the response it walked away from: its identity goes everywhere.
        assert (await client.delete('/v1/responses/' + done)).status == 200
        assert not [row for row in store._conn.execute(
            "SELECT request_key FROM response_keys WHERE request_key LIKE '%gone-done'")]
        # A deleted key is still never permission to run again.
        retry = await client.post('/v1/responses', json={'input': 'gone-done', 'stream': True},
                                  headers={'Idempotency-Key': 'gone-done'})
        assert retry.status == 409
        # Running work keeps its identity: delete drops the snapshot, never the running key.
        assert (await client.delete('/v1/responses/' + busy)).status == 200
        keys = [row[0].rsplit(':', 1)[1] for row in store._conn.execute('SELECT request_key FROM response_keys')]
        assert keys == ['held'], keys
        hold.set()
        await _await_terminal(owner, 2)
    assert calls == ['gone-done', 'held']


@pytest.mark.asyncio
async def test_successful_retry_repairs_disconnected_snapshot(api, owner, fast_keepalive):
    """A stream that left after ``response.created`` leaves an ``incomplete`` snapshot; a retry that
    observes the committed result replaces it, so GET and ``previous_response_id`` see the answer."""
    calls, release = [], asyncio.Event()
    store = api._current_response_store()
    async with TestClient(TestServer(_app(api, owner, calls, release))) as client:
        rid = await _disconnect_mid_stream(client, store, 'gone-repair')
        release.set()
        await _await_terminal(owner, 1)
        retry = await client.post('/v1/responses', json={'input': 'gone-repair', 'stream': True},
                                  headers={'Idempotency-Key': 'gone-repair'})
        wire = await retry.text()
        assert 'response.completed' in wire and 'answer gone-repair' in wire
        saved = store.get(rid)
        assert saved['response']['status'] == 'completed'
        assert saved['conversation_history'][-1] == {'role': 'assistant', 'content': 'answer gone-repair'}
    assert calls == ['gone-repair']


@pytest.mark.asyncio
async def test_a_failed_retry_leaves_the_disconnected_snapshot_unclaimed(api, owner, fast_keepalive, monkeypatch):
    """Only an observation of the committed result finalizes a provisional snapshot: a retry whose
    own observation fails does not record the response as failed."""
    from gateway import session_api_turn
    from hermes_state_runtime import RuntimeStoreError
    calls, release = [], asyncio.Event()
    store = api._current_response_store()
    async with TestClient(TestServer(_app(api, owner, calls, release))) as client:
        rid = await _disconnect_mid_stream(client, store, 'gone-unclaimed')
        release.set()
        await _await_terminal(owner, 1)

        async def unavailable(*args, **kwargs):
            raise RuntimeStoreError('storage_unavailable')
        monkeypatch.setattr(session_api_turn, 'observe_api_turn', unavailable)
        retry = await client.post('/v1/responses', json={'input': 'gone-unclaimed', 'stream': True},
                                  headers={'Idempotency-Key': 'gone-unclaimed'})
        assert 'response.failed' in await retry.text()
        assert store.get(rid)['response']['status'] == 'incomplete'
    assert calls == ['gone-unclaimed']


_SNAPSHOT_STATUSES = ('missing', 'in_progress', 'incomplete', 'failed', 'completed')


@pytest.mark.asyncio
@pytest.mark.parametrize('existing', _SNAPSHOT_STATUSES)
@pytest.mark.parametrize('outcome', ('committed', 'observer_failure', 'disconnect'))
@pytest.mark.parametrize('pointer', ('missing', 'newer'))
async def test_a_replay_supersedes_only_what_the_committed_result_may_replace(api, existing, outcome, pointer):
    """The replay rule for every stored snapshot: an observation of the committed result replaces
    anything but a completed snapshot; a replay's own failure or disconnect replaces nothing. The
    conversation pointer, if evicted, is restored to the snapshot that stands; a newer one stays."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from gateway.platforms.api_server_openai_routes import _ResponsesStream
    store = api._current_response_store()
    rid = 'resp_' + existing + '_' + outcome
    before = None
    if existing != 'missing':
        before = {'response': {'id': rid, 'object': 'response', 'status': existing, 'output': []},
                  'conversation_history': [{'role': 'user', 'content': 'work'}], 'instructions': None,
                  'session_id': 's'}
        store.put(rid, before)
    if pointer == 'newer':
        store.set_conversation('room', 'resp_newer')
    stream = _ResponsesStream(api, SimpleNamespace(write=AsyncMock()), response_id=rid, model='m', created_at=0,
                              conversation_history=[], user_message='work', instructions=None, conversation='room',
                              store=True, session_id='s', terminal_replay=True)
    if outcome == 'committed':
        stream.persist_snapshot({'id': rid, 'object': 'response', 'status': 'completed', 'output': []},
                                history=[{'role': 'user', 'content': 'work'}, {'role': 'assistant', 'content': 'answer'}],
                                committed=True)
    elif outcome == 'observer_failure':
        stream.agent_error = 'storage_unavailable'
        await stream.emit_failed()
    else:
        stream.final_text_parts = ['partial']
        stream.persist_incomplete_if_needed()
    saved = store.get(rid)
    if outcome == 'committed' and existing != 'completed':
        assert saved['response']['status'] == 'completed'
        assert saved['conversation_history'][-1] == {'role': 'assistant', 'content': 'answer'}
    else:
        assert saved == before
    assert store.get_conversation('room') == ('resp_newer' if pointer == 'newer' else rid if saved else None)


@pytest.mark.asyncio
async def test_a_replay_restores_a_conversation_pointer_the_lru_evicted(api, owner):
    """The LRU evicted ``idem:A`` and B's snapshot, and with it ``room``'s pointer. A's replay
    restores the pointer to A's completed snapshot, so the next turn of ``room`` gets A's history."""
    from gateway import session_api_turn as turns
    from tests.gateway.test_responses_identity_retention import _app as plain_app
    calls, histories = [], []
    app = plain_app(api, owner, calls)
    handle = owner.runner._handle_message

    async def record(event):
        histories.append(list(turns.api_execution.get()['history']))
        return await handle(event)
    owner.runner._handle_message = record
    store = api._current_response_store()
    store._max_size = 4
    async with TestClient(TestServer(app)) as client:
        ids = {}
        for key in ('A', 'B'):
            response = await client.post('/v1/responses', json={'input': key, 'conversation': 'room'},
                                         headers={'Idempotency-Key': key})
            ids[key] = (await response.json())['id']
        for unrelated in ('x1', 'x2'):
            store.get(ids['A'])
            store.put(unrelated, {'response': {'id': unrelated, 'status': 'completed'}})
        replay_key = next(row[0] for row in store._conn.execute('SELECT request_key FROM response_keys')
                          if row[0].startswith('idem:') and row[0].endswith(':A'))
        assert store.get(replay_key) is None and store.get(ids['B']) is None
        assert store.get(ids['A'])['response']['status'] == 'completed' and store.get_conversation('room') is None
        retry = await client.post('/v1/responses', json={'input': 'A', 'conversation': 'room', 'stream': True},
                                  headers={'Idempotency-Key': 'A'})
        assert 'response.completed' in await retry.text()
        assert store.get_conversation('room') == ids['A']
        follow = await client.post('/v1/responses', json={'input': 'next', 'conversation': 'room'})
        assert follow.status == 200, await follow.text()
    assert calls == ['A', 'B', 'next']
    assert histories[-1][-1] == {'role': 'assistant', 'content': 'answer A'}
