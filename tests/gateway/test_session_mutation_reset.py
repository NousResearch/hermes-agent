"""Reset commits the prepared physical target without changing logical ownership."""
import asyncio
import json
import sqlite3

import pytest
from websockets.asyncio.client import connect
from tests.gateway.test_native_http_auth import daemon, ticket


async def reset_round(home, descriptor, previous=None):
    async with connect(descriptor['api_origin'].replace('http:', 'ws:') + '/api/ws', subprotocols=[
            'hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket(home, descriptor, 'interactive')]) as ws:
        async def rpc(method, **params):
            await ws.send(json.dumps({'id': 1, 'method': method, 'params': params}))
            while True:
                result = json.loads(await ws.recv())
                if result.get('id') == 1:
                    return result
        if previous is None:
            created = await rpc('session.create', request_id='reset-owner', source='gui')
            sid = created['result']['session_id']
            params = dict(session_id=sid, request_id='reset', expected_revision=0,
                          expected_generation=0, operation='reset', payload={})
        else:
            params, receipt = previous
            sid = params['session_id']
        response = await rpc('session.mutate', **params)
        assert 'result' in response, response
        if previous:
            assert response['result'] == receipt
        else:
            receipt = response['result']
        assert receipt['session_id'] == sid
        assert receipt['target_session_id'] != sid
        resumed = await rpc('session.resume', session_id=sid)
        assert 'result' in resumed, resumed
        stale = await rpc('session.mutate', **(params | {'request_id': 'stale-reset'}))
        assert stale['error']['message'] == 'revision_conflict'
        return params, receipt


@pytest.mark.linux_only
def test_reset_receipt_and_physical_target_survive_native_owner_restart(tmp_path):
    with daemon(tmp_path) as (home, descriptor):
        previous = asyncio.run(reset_round(home, descriptor))
        sid = previous[0]['session_id']
        target = previous[1]['target_session_id']
        with sqlite3.connect(home / 'state.db') as db:
            root = db.execute('SELECT runtime_revision,runtime_generation,end_reason FROM sessions WHERE id=?', (sid,)).fetchone()
            assert root == (previous[1]['revision'], previous[1]['execution_generation'], 'session_reset')
            assert db.execute('SELECT parent_session_id FROM sessions WHERE id=?', (target,)).fetchone()[0] == sid
            assert db.execute('SELECT COUNT(*) FROM messages WHERE session_id=?', (target,)).fetchone()[0] == 0
    with daemon(tmp_path) as (home, descriptor):
        assert asyncio.run(reset_round(home, descriptor, previous)) == previous
    print(json.dumps({'native_reset_restart': previous[1]}))
