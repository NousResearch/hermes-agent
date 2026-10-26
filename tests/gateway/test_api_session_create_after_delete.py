"""``POST /api/sessions`` with the explicit id of a deleted session is refused when it is created.

A deleted id stays retired for good, so every turn on it is refused. Recreating the row beside the
fence used to answer 201: the session listed fine, then refused every turn. The create now says so
at once (409 ``session_deleted``), and the client mints a new id. An explicit id is the client's
own name for the conversation, so it is never remapped to another session behind its back.
"""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

_KEY = 'create-after-delete-key'
_AUTH = {'Authorization': 'Bearer ' + _KEY}


@pytest.mark.asyncio
@pytest.mark.parametrize('past_horizon', [False, True], ids=['within-horizon', 'past-horizon'])
async def test_creating_a_session_with_a_deleted_id_is_refused_by_name(api, owner, past_horizon):
    from hermes_state_mutation_retirement import prune_retired_tombstones
    api._api_key = _KEY
    app = web.Application()
    app.router.add_post('/api/sessions', api._handle_create_session)
    app.router.add_delete('/api/sessions/{session_id}', api._handle_delete_session)
    async with TestClient(TestServer(app)) as client:
        created = await client.post('/api/sessions', json={'id': 'client-conv'}, headers=_AUTH)
        assert created.status == 201, await created.text()
        deleted = await client.delete('/api/sessions/client-conv', headers=_AUTH)
        assert deleted.status == 200, await deleted.text()
        if past_horizon:
            owner.db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
        again = await client.post('/api/sessions', json={'id': 'client-conv'}, headers=_AUTH)
        payload = await again.json()
        assert again.status == 409 and payload['error']['code'] == 'session_deleted', payload
        assert owner.db.get_session('client-conv') is None, 'the deleted row was recreated beside its fence'
        fresh = await client.post('/api/sessions', json={'id': 'client-conv-2'}, headers=_AUTH)
        assert fresh.status == 201, await fresh.text()
