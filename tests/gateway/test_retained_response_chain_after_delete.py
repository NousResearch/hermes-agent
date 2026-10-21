"""A retained Responses chain whose transcript was deleted, continued with a fresh Idempotency-Key,
moves to a new generation: it is neither refused forever (``not_found``) nor allowed to recreate
the deleted transcript id. The history comes from the response store, so nothing is lost.
"""
import json
import sqlite3

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

_API_KEY = 'test-key'
_AUTH = {'Authorization': 'Bearer ' + _API_KEY}


@pytest.mark.asyncio
@pytest.mark.parametrize('path, continuation', [
    ('/v1/responses', 'previous_response_id'),
    ('/v1/responses', 'conversation'),
    ('/v1/runs', 'previous_response_id'),
])
async def test_a_deleted_chain_continues_on_a_new_generation(api, owner, path, continuation):
    from gateway.session_authority import SessionAuthority
    from gateway.session_results import execution_result
    from hermes_state_mutation_retirement import prune_retired_tombstones
    from hermes_state_runtime import begin_runtime_epoch

    async def handle(event):
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    owner.runner._handle_message = handle
    api._api_key = _API_KEY
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    app.router.add_post('/v1/runs', api._handle_runs)
    app.router.add_delete('/api/sessions/{session_id}', api._handle_delete_session)
    async with TestClient(TestServer(app)) as client:
        opened = await client.post('/v1/responses', json={'input': 'remember 42', 'conversation': 'project'},
                                   headers={**_AUTH, 'Idempotency-Key': 'first'})
        assert opened.status == 200, await opened.text()
        response_id = json.loads(await opened.text())['id']
        deleted_sid = api._current_response_store().get(response_id)['session_id']
        assert (await client.delete('/api/sessions/' + deleted_sid, headers=_AUTH)).status == 200
        owner.db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
        # Past the horizon is a later process: a fresh authority over the same store.
        owner.runner.session_authority = SessionAuthority(
            owner.runner, profile_id='default', instance_id='later', db=owner.db,
            epoch=begin_runtime_epoch(owner.db, instance_id='later'))
        target = response_id if continuation == 'previous_response_id' else 'project'
        resumed = await client.post(path, json={'input': 'what number?', continuation: target},
                                    headers={**_AUTH, 'Idempotency-Key': 'fresh'})
        body = await resumed.text()
        assert resumed.status in (200, 202), f'{resumed.status}: {body}'
        assert owner.db.get_session(deleted_sid) is None, 'the deleted transcript id was recreated'


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['/v1/responses', '/v1/runs'])
async def test_a_busy_store_during_the_chain_lookup_is_retryable(api, owner, monkeypatch, path):
    """The chain's generation lookup reads the owning store before the response starts: a busy
    store is the retryable JSON 503 ``storage_unavailable``, not aiohttp's plain-text 500."""
    from gateway import session_api
    from gateway.session_results import execution_result

    async def handle(event):
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    owner.runner._handle_message = handle
    api._api_key = _API_KEY
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    app.router.add_post('/v1/runs', api._handle_runs)
    async with TestClient(TestServer(app)) as client:
        opened = await client.post('/v1/responses', json={'input': 'hello'},
                                   headers={**_AUTH, 'Idempotency-Key': 'first'})
        assert opened.status == 200, await opened.text()
        response_id = json.loads(await opened.text())['id']

        def busy(*args, **kwargs):
            raise sqlite3.OperationalError('database is locked')

        monkeypatch.setattr(session_api, 'implicit_api_session', busy)
        resumed = await client.post(path, json={'input': 'next', 'previous_response_id': response_id},
                                    headers={**_AUTH, 'Idempotency-Key': 'second'})
        assert resumed.status == 503, (resumed.status, await resumed.text())
        assert resumed.headers.get('Retry-After') == '1'
        assert (await resumed.json())['error']['code'] == 'storage_unavailable'
