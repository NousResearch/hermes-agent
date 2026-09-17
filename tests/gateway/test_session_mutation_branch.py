"""Branches are independent frozen local owners, never route aliases."""
import asyncio
import json
import sqlite3

import pytest
from websockets.asyncio.client import connect
from tests.gateway.test_native_http_auth import daemon, ticket


async def branch_round(home, descriptor, previous=None):
    async with connect(descriptor['api_origin'].replace('http:', 'ws:') + '/api/ws', subprotocols=[
            'hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket(home, descriptor, 'interactive')]) as ws:
        async def rpc(method, **params):
            await ws.send(json.dumps({'id': 1, 'method': method, 'params': params}))
            while True:
                result = json.loads(await ws.recv())
                if result.get('id') == 1:
                    return result
        if previous is None:
            created = await rpc('session.create', request_id='branch-owner', source='gui', toolsets=[])
            sid = created['result']['session_id']
            params = dict(session_id=sid, request_id='branch', expected_revision=0,
                          expected_generation=0, operation='branch', payload={})
        else:
            params, receipt = previous
        response = await rpc('session.mutate', **params)
        assert 'result' in response, response
        if previous:
            assert response['result'] == receipt
        receipt = response['result']
        child = receipt['branched_session_id']
        assert child != params['session_id']
        if previous is None:
            # The branching viewer owns the child it just created: a submit on the
            # same connection is admitted without an explicit resume first (create parity).
            admitted = await rpc('prompt.submit', session_id=child, input_id='first-child-input', text='FIRST_CHILD_INPUT')
            assert admitted.get('result', {}).get('status') == 'queued', admitted
        for sid in (child, params['session_id']):
            resumed = await rpc('session.resume', session_id=sid)
            assert 'result' in resumed, resumed
        stale = await rpc('session.mutate', **(params | {'request_id': 'stale-branch'}))
        assert stale['error']['message'] == 'revision_conflict'
        changed = await rpc('session.mutate', **(params | {'payload': {'title': 'changed'}}))
        assert changed['error']['message'] == 'admission_conflict'
        return params, receipt


@pytest.mark.linux_only
def test_branch_route_and_frozen_policy_survive_owner_restart(tmp_path):
    with daemon(tmp_path) as (home, descriptor):
        previous = asyncio.run(branch_round(home, descriptor))
        sid = previous[0]['session_id']
        child = previous[1]['branched_session_id']
        with sqlite3.connect(home / 'state.db') as db:
            policies = [json.loads(db.execute('SELECT value FROM state_meta WHERE key=?',
                ('gateway.local_policy.v1:' + s,)).fetchone()[0]) for s in (sid, child)]
            assert policies[0]['policy'] == policies[1]['policy']
            assert policies[0]['principal_id'] == policies[1]['principal_id']
            assert policies[0]['route'] != policies[1]['route']
            assert db.execute('SELECT parent_session_id FROM sessions WHERE id=?', (child,)).fetchone()[0] == sid
    with daemon(tmp_path) as (home, descriptor):
        assert asyncio.run(branch_round(home, descriptor, previous)) == previous
