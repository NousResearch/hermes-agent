"""Target-authorized succession links homes without merging unrelated room origins."""
import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.hosted_room_peer import decode_room_grant
from gateway.platforms.api_server_run_authority import room_authority, room_run_scope
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tests.gateway.test_api_server_room_cancellation import _adapter, _headers, _invitation
from tests.gateway.test_room_cancellation_retirement_http import app


def successor_body(**changes):
    return {'room_id': 'room-1', 'home_install_id': 'successor', 'authority_gateway_id': 'successor',
            'authority_epoch': 2, 'member_id': 'member-1', 'previous_authority': {
                'home_install_id': 'home', 'authority_gateway_id': 'home', 'authority_epoch': 1}, **changes}


async def invite(cli, body, authorization='Bearer test-room-key'):
    return await cli.post('/v1/room-members/invitations', headers={'Authorization': authorization}, json=body)


@pytest.mark.asyncio
async def test_changed_home_compacts_with_explicit_lineage_and_survives_restart(tmp_path):
    path = tmp_path / 'runs.db'
    adapter = _adapter(path)
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            claims = decode_room_grant(adapter._room_grant_secret(), grant, permission='status')
            for index in range(64):
                body['hosted_room_dispatch']['task_id'] = f'task-{index}'
                headers = {**_headers(grant), 'Idempotency-Key': f'room:task-{index}:1'}
                assert (await cli.post('/v1/runs/stop', headers=headers, json=body)).status == 200
            store = adapter._run_idempotency_store
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 64
            # Signed peer authority cannot authorize a new lineage on the target's behalf.
            assert (await invite(cli, successor_body(), f'HermesRoom {grant}')).status == 401
            assert (await invite(cli, successor_body())).status == 201
            assert (await invite(cli, successor_body())).status == 201
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 0
            assert store._conn.execute('SELECT COUNT(*) FROM run_room_authorities').fetchone()[0] == 1
            store.close()
            store = adapter._run_idempotency_store = RunIdempotencyStore(str(path))
            for index in range(64):
                assert store.reserve(room_run_scope(claims), f'room:task-{index}:1', 'late', 'never-start',
                    {'status': 'queued'}, room_authority=room_authority(claims)) == ('authority_retired', None)
            # A second actual home move uses the retained successor, never the original host.
            third = successor_body(home_install_id='third', authority_gateway_id='third', authority_epoch=3,
                previous_authority={'home_install_id': 'successor', 'authority_gateway_id': 'successor', 'authority_epoch': 2})
            assert (await invite(cli, third)).status == 201
            assert (await invite(cli, successor_body(authority_epoch=4))).status == 400
            assert store._conn.execute('SELECT COUNT(*) FROM run_room_authorities').fetchone()[0] == 1
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_unrelated_origin_collision_does_not_compact_or_merge(tmp_path):
    adapter = _adapter(tmp_path / 'runs.db')
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            assert (await cli.post('/v1/runs/stop', headers=_headers(grant), json=body)).status == 200
            unrelated = successor_body()
            unrelated.pop('previous_authority')
            assert (await invite(cli, unrelated)).status == 400
            store = adapter._run_idempotency_store
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 1
            assert store._conn.execute('SELECT COUNT(*) FROM run_room_authorities').fetchone()[0] == 1
            # Refusing the collision also preserves the original grant's live reservation.
            assert (await cli.post('/v1/runs/stop', headers=_headers(grant), json=body)).status == 200
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 1
            assert store._conn.execute('SELECT COUNT(*) FROM run_room_authority_aliases').fetchone()[0] == 0
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy', [False, True])
async def test_unlinked_collision_refused_after_restart_or_with_legacy_reservation(tmp_path, legacy):
    from gateway import hosted_rooms
    path = tmp_path / 'runs.db'
    adapter = _adapter(path)
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            await _invitation(cli)
            if legacy:
                adapter._run_idempotency_store._conn.execute('DELETE FROM run_room_namespaces')
                adapter._run_idempotency_store._conn.commit()
            else:
                with hosted_rooms._transaction(hosted_rooms.default_db_path(), immediate=True) as conn:
                    conn.execute('DELETE FROM hosted_room_peer_reservations')
            adapter._run_idempotency_store.close()
            adapter._run_idempotency_store = RunIdempotencyStore(str(path))
            unrelated = successor_body()
            unrelated.pop('previous_authority')
            assert (await invite(cli, unrelated)).status == 400
            assert (await invite(cli, successor_body())).status == 201
    finally:
        adapter._run_idempotency_store.close()


def test_explicit_binding_cannot_absorb_a_preexisting_unrelated_origin(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        first, unrelated = ('first', 1, 'home'), ('unrelated', 2, 'other')
        store.observe_room_authority('first-scope', first)
        store.observe_room_authority('other-scope', unrelated)
        with pytest.raises(ValueError, match='another room origin'):
            store.observe_room_authority('new-scope', ('unrelated', 3, 'other'), first, 'home')
        assert store._conn.execute('SELECT COUNT(*) FROM run_room_authority_aliases').fetchone()[0] == 0
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('previous', [
    {'home_install_id': 'forged', 'authority_gateway_id': 'home', 'authority_epoch': 1},
    {'home_install_id': 'home', 'authority_gateway_id': 'forged', 'authority_epoch': 1},
    {'home_install_id': 'home', 'authority_gateway_id': 'home', 'authority_epoch': 2}])
async def test_invalid_predecessor_does_not_advance_authority(tmp_path, previous):
    adapter = _adapter(tmp_path / 'runs.db')
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            assert (await invite(cli, successor_body(previous_authority=previous))).status == 400
            response = await cli.post('/v1/runs/stop', headers=_headers(grant), json=body)
            assert response.status == 200 and (await response.json())['admission_cancelled']
    finally:
        adapter._run_idempotency_store.close()


def test_legacy_watermark_migrates_without_guessing_opaque_receipt_identity(tmp_path):
    import sqlite3
    path = tmp_path / 'legacy.db'
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE run_room_authorities (authority_key TEXT PRIMARY KEY,
            authority_epoch INTEGER NOT NULL, gateway_id TEXT NOT NULL, retired_through INTEGER NOT NULL DEFAULT 0)''')
        conn.execute("INSERT INTO run_room_authorities VALUES('old-home-key',1,'home',0)")
    store = RunIdempotencyStore(str(path))
    try:
        old = ('old-home-key', 1, 'home')
        for scope in ('known-old', 'opaque-old'):
            store.reserve(scope, 'cancelled', '', scope, {'status': 'cancelled', 'admission_cancelled': True},
                          cancel_if_missing=True)
        assert store.observe_room_authority('known-old', old)
        assert store.observe_room_authority('new-scope', ('new-home-key', 2, 'successor'), old, 'home')
        assert store._conn.execute('SELECT scope,room_authority_key FROM run_idempotency').fetchall() == [('opaque-old', None)]
        assert store.reserve('known-old', 'cancelled', '', 'late', {'status': 'queued'},
                             room_authority=old) == ('authority_retired', None)
        assert store.lookup('opaque-old', 'cancelled', '')[0] == 'reused'
    finally:
        store.close()
