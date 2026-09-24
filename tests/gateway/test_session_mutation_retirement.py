"""Native deletion retires routing and keeps only exact authorized retries."""
import asyncio
import json
import sqlite3

import httpx
import pytest
from websockets.asyncio.client import connect
from tests.gateway.test_native_http_auth import daemon, ticket, headers


async def create(home, descriptor):
    async with connect(descriptor['api_origin'].replace('http:', 'ws:') + '/api/ws',
        subprotocols=['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket(home, descriptor, 'interactive')]) as ws:
        await ws.send(json.dumps({'id': 1, 'method': 'session.create', 'params': {
            'request_id': 'retirement', 'source': 'gui'}}))
        while True:
            response = json.loads(await ws.recv())
            if response.get('id') == 1:
                assert 'result' in response, response
                return response['result']['session_id']


@pytest.mark.linux_only
def test_native_delete_exact_retry_survives_owner_restart(tmp_path):
    with daemon(tmp_path) as (home, descriptor), httpx.Client(base_url=descriptor['api_origin'], trust_env=False) as client:
        sid = asyncio.run(create(home, descriptor))
        path = '/api/sessions/' + sid
        snapshot = client.get(path + '/mutation-snapshot', headers=headers(ticket(home, descriptor))).json()
        params = dict(request_id='delete', expected_revision=snapshot['runtime_revision'],
                      expected_generation=snapshot['runtime_generation'])
        deleted = client.delete(path, params=params, headers=headers(ticket(home, descriptor)))
        assert deleted.status_code == 200, deleted.text
        receipt = deleted.json()
        assert client.get(path, headers=headers(ticket(home, descriptor))).status_code == 404
        with sqlite3.connect(home / 'state.db') as db:
            assert not any(json.loads(row[0])['session_id'] == sid for row in db.execute('SELECT entry_json FROM gateway_routing'))
        assert client.delete(path, params=params, headers=headers(ticket(home, descriptor))).json() == receipt
    with daemon(tmp_path) as (home, descriptor), httpx.Client(base_url=descriptor['api_origin'], trust_env=False) as client:
        retry = client.delete(path, params=params, headers=headers(ticket(home, descriptor)))
        assert retry.status_code == 200 and retry.json() == receipt, retry.text
        changed = client.delete(path, params=params | {'expected_revision': 99}, headers=headers(ticket(home, descriptor)))
        assert changed.status_code == 409 and 'admission_conflict' in changed.text
        fresh = client.delete(path, params=params | {'request_id': 'different'}, headers=headers(ticket(home, descriptor)))
        assert fresh.status_code != 200
        foreign = client.delete(path, params=params, headers={'Authorization': 'Bearer normal-http-owner'})
        assert foreign.status_code != 200
    print(json.dumps({'native_delete_restart': receipt, 'changed_retry': changed.json(),
                      'fresh_request_status': fresh.status_code, 'foreign_status': foreign.status_code}))


def test_route_retirement_rolls_back_with_receipt_and_rejects_stale_mirror(tmp_path):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore, SessionSource
    from hermes_state import SessionDB
    import hermes_state_runtime as rt
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        store._db = db
        source = SessionSource(platform=Platform.TELEGRAM, chat_id='owned')
        entry = store.get_or_create_session(source)
        sid = entry.session_id
        if db.get_session(sid) is None:
            db.create_session(sid, source='telegram')
        serialized = json.dumps(entry.to_dict())
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        args = dict(epoch=epoch, principal_id='human', session_id=sid, request_id='delete',
                    expected_revision=0, expected_generation=0, operation='delete', payload={})
        db._execute_write(lambda conn: conn.execute("CREATE TRIGGER reject_receipt BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.mutation.v1.%' BEGIN SELECT RAISE(ABORT,'receipt unavailable'); END"))
        with pytest.raises(sqlite3.IntegrityError, match='receipt unavailable'):
            rt.mutate_runtime_session(db, **args)
        assert db.get_session(sid) is not None
        assert db.load_gateway_routing_entries(scope=store._routing_scope())
        db._execute_write(lambda conn: conn.execute('DROP TRIGGER reject_receipt'))
        rt.mutate_runtime_session(db, **args)
        assert not db.load_gateway_routing_entries(scope=store._routing_scope())
        db.save_gateway_routing_entry(entry.session_key, serialized, scope=store._routing_scope())
        db.replace_gateway_routing_entries({entry.session_key: serialized}, scope=store._routing_scope())
        assert not db.load_gateway_routing_entries(scope=store._routing_scope())
        # The pre-commit mirror remains on disk as in a crash before publication.
        restored = SessionStore(store.sessions_dir, GatewayConfig())
        restored._db = db
        assert restored.lookup_by_session_id(sid) is None
