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


@pytest.mark.platforms("linux")
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
        # Without this principal's exact receipt, a delete of the gone id is main's idempotent
        # ``already_absent`` (Desktop ghost-row contract, #48641): never the receipt, never a write.
        absent = {'ok': True, 'already_absent': True}
        fresh = client.delete(path, params=params | {'request_id': 'different'}, headers=headers(ticket(home, descriptor)))
        assert fresh.status_code == 200 and fresh.json() == absent, fresh.text
        # The dashboard login is the owner, so its exact retry replays the owner's receipt.
        login = client.delete(path, params=params, headers={'Authorization': 'Bearer normal-http-owner'})
        assert login.status_code == 200 and login.json() == receipt, login.text
    print(json.dumps({'native_delete_restart': receipt, 'changed_retry': changed.json(),
                      'fresh_request_status': fresh.status_code, 'login_status': login.status_code}))


def test_route_retirement_rolls_back_with_receipt_and_rejects_stale_mirror(tmp_path):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore, SessionSource
    from persistence.authority.runtime import Delete
    from persistence.core.errors import PermanentStoreError
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_holder import raw_holder
    from tests.fixtures.storage_read import session_row
    from tests.fixtures.storage_seed import seed_session
    # The stores below resolve the active profile's state.db, the one this handle opens.
    with open_persistence(profile_scope()) as db:
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        source = SessionSource(platform=Platform.TELEGRAM, chat_id='owned')
        entry = store.get_or_create_session(source)
        sid = entry.session_id
        if session_row(db, sid) is None:
            seed_session(db, sid, 'telegram')
        serialized = json.dumps(entry.to_dict())
        epoch = db.runtime.begin_epoch(instance_id='owner')
        args = dict(principal_id='human', session_id=sid, request_id='delete',
                    expected_revision=0, expected_generation=0, command=Delete())
        raw_holder(db)._execute_write(lambda conn: conn.execute("CREATE TRIGGER reject_receipt BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.mutation.v1.%' BEGIN SELECT RAISE(ABORT,'receipt unavailable'); END"))
        with pytest.raises(PermanentStoreError, match='receipt unavailable') as refused:
            db.runtime.mutate(epoch, **args, owner=None)
        assert isinstance(refused.value.__cause__, sqlite3.IntegrityError)
        assert session_row(db, sid) is not None
        assert db.routing.load_routing_entries(routing_scope=store._routing_scope())
        raw_holder(db)._execute_write(lambda conn: conn.execute('DROP TRIGGER reject_receipt'))
        db.runtime.mutate(epoch, **args, owner=None)
        assert not db.routing.load_routing_entries(routing_scope=store._routing_scope())
        db.routing.upsert_routing_entry(entry.session_key, serialized, routing_scope=store._routing_scope())
        db.routing.replace_routing_entries({entry.session_key: serialized}, routing_scope=store._routing_scope())
        assert not db.routing.load_routing_entries(routing_scope=store._routing_scope())
        # The pre-commit mirror remains on disk as in a crash before publication.
        restored = SessionStore(store.sessions_dir, GatewayConfig())
        assert restored.lookup_by_session_id(sid) is None
