"""Fresh target-owner authorization retires expired scopes without reopening execution."""
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_rooms
from gateway.hosted_room_peer import decode_room_grant, HostedRoomGrantError
from gateway.platforms.api_server_run_authority import room_authority
from tests.gateway.test_api_server_room_cancellation import _adapter, _headers, _invitation
from tests.gateway.test_room_authority_lineage import invite, successor_body
from tests.gateway.test_room_cancellation_retirement_http import app


def retirement_app(adapter):
    result = app(adapter)
    result.router.add_get('/v1/room-members/capabilities', adapter._handle_room_member_capabilities)
    result.router.add_post('/v1/room-members/grants/refresh', adapter._handle_room_member_grant_refresh)
    return result


def reservations():
    with hosted_rooms._transaction(hosted_rooms.default_db_path()) as conn:
        return [tuple(row) for row in conn.execute('SELECT * FROM hosted_room_peer_reservations')]


@pytest.mark.asyncio
async def test_owner_recovers_expired_retirement_and_lost_reply_without_reopening_room(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path / 'runs.db')
    try:
        async with TestClient(TestServer(retirement_app(adapter))) as cli:
            grant, body = await _invitation(cli)
            assert (await cli.post('/v1/runs/stop', headers=_headers(grant), json=body)).status == 200
            claims = decode_room_grant(adapter._room_grant_secret(), grant, permission='status')
            monkeypatch.setattr(time, 'time', lambda: claims['status_expires_at'] + 1)
            assert (await cli.post('/v1/room-members/grants/revoke', headers=_headers(grant),
                                   json={'retire_authority': True})).status == 401
            prior = reservations()
            store = adapter._run_idempotency_store
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 1
            recovery_body = {key: claims[key] for key in (
                'room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch', 'member_id')}
            recovery_body['retirement_only'] = True
            assert (await invite(cli, recovery_body, f'HermesRoom {grant}')).status == 401
            response = await invite(cli, recovery_body)
            assert response.status == 201
            narrow = (await response.json())['grant']
            narrow_claims = decode_room_grant(adapter._room_grant_secret(), narrow, permission='retire')
            assert set(narrow_claims['permissions']) == {'status', 'retire'}
            assert reservations() == prior
            for permission in ('dispatch', 'approve', 'stop'):
                with pytest.raises(HostedRoomGrantError):
                    decode_room_grant(adapter._room_grant_secret(), narrow, permission=permission)
            probe = await cli.get('/v1/room-members/capabilities', headers=_headers(narrow))
            assert probe.status == 200 and (await probe.json())['retirement_only']
            for path in ('/v1/runs', '/v1/runs/stop', '/v1/room-members/grants/refresh'):
                assert (await cli.post(path, headers=_headers(narrow), json=body if '/runs' in path else {})).status == 401
            retired = await cli.post('/v1/room-members/grants/revoke', headers=_headers(narrow),
                                     json={'retire_authority': True})
            assert retired.status == 200 and (await retired.json())['authority_retired']
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 0
            # A lost retirement reply is recoverable even after its token has expired.
            monkeypatch.setattr(time, 'time', lambda: narrow_claims['status_expires_at'] + 1)
            prior = reservations()
            response = await invite(cli, recovery_body)
            assert response.status == 201 and reservations() == prior
            retry = (await response.json())['grant']
            assert (await cli.get('/v1/room-members/capabilities', headers=_headers(retry))).status == 200
            assert (await cli.post('/v1/room-members/grants/revoke', headers=_headers(retry),
                                   json={'retire_authority': True})).status == 200
            assert not store.accepts_room_authority(room_authority(claims))
            recovery_body.pop('retirement_only')
            assert (await invite(cli, recovery_body)).status == 400
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_retirement_of_predecessor_never_retires_replacement_owner(tmp_path):
    adapter = _adapter(tmp_path / 'runs.db')
    try:
        async with TestClient(TestServer(retirement_app(adapter))) as cli:
            old_grant, _ = await _invitation(cli)
            response = await invite(cli, successor_body())
            new_grant = (await response.json())['grant']
            new_claims = decode_room_grant(adapter._room_grant_secret(), new_grant, permission='status')
            prior = reservations()
            wrong = successor_body(retirement_only=True, authority_gateway_id='forged')
            wrong.pop('previous_authority')
            assert (await invite(cli, wrong)).status == 400 and reservations() == prior
            # A stale broad bearer cannot retire a successor; only a new owner-approved
            # retirement grant can discharge the predecessor's separate obligation.
            assert (await cli.post('/v1/room-members/grants/revoke', headers=_headers(old_grant),
                                   json={'retire_authority': True})).status == 401
            old = successor_body(home_install_id='home', authority_gateway_id='home', authority_epoch=1,
                                 retirement_only=True)
            old.pop('previous_authority')
            response = await invite(cli, old)
            assert response.status == 201 and reservations() == prior
            narrow = (await response.json())['grant']
            assert (await cli.post('/v1/room-members/grants/revoke', headers=_headers(narrow),
                                   json={'retire_authority': True})).status == 200
            assert reservations() == prior
            assert adapter._run_idempotency_store.accepts_room_authority(room_authority(new_claims))
            assert (await cli.get('/v1/room-members/capabilities', headers=_headers(new_grant))).status == 200
    finally:
        adapter._run_idempotency_store.close()
