"""Issued setup grants survive a lost reply and cannot escape their actor/body."""
import asyncio
import json
import time

from gateway import hosted_rooms
from gateway.platforms.api_server_room_grants import _grant_db
from gateway.session_controls import AuthorityConnection
from tests.gateway.test_session_group_peers import gateway, call


def test_setup_invitation_replay_is_frozen_and_revocable_after_lost_reply(gateway, monkeypatch):
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'http://127.0.0.1:9')
    params = dict(room_id='setup', home_install_id='install:home', authority_gateway_id='install:home',
        authority_epoch=1, member_id='peer', request_id='desktop-issuance-1', requested_at=time.time(),
        ttl_seconds=60, status_ttl_seconds=3600)
    first = asyncio.run(call(gateway.owner, 'groups.peer.invite', **params))
    assert isinstance(first, dict), first
    again = asyncio.run(call(gateway.owner, 'groups.peer.invite', **params))
    assert again == first
    assert asyncio.run(call(gateway.owner, 'groups.peer.invite', **(params | {'member_id': 'replacement'}))) == 'idempotency_conflict'
    other = AuthorityConnection(gateway.authority, object(), {'user_id': 'other'}, operator=True)
    assert asyncio.run(call(other, 'groups.peer.invite', **params)) == 'idempotency_conflict'
    assert asyncio.run(call(gateway.owner, 'groups.peer.revoke', grant=again['grant'])) == {'revoked': True}
    # Replay never remints authority after cleanup, even though the response was lost.
    assert asyncio.run(call(gateway.owner, 'groups.peer.invite', **params)) == first
    from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret
    claims = decode_room_grant(gateway_room_grant_secret(), first['grant'], permission='status')
    assert hosted_rooms.room_grant_is_revoked(_grant_db(gateway.adapter), claims=claims)
    assert asyncio.run(call(gateway.owner, 'groups.peer.invite', **(params | {
        'request_id': 'desktop-expired-1', 'requested_at': time.time() - 600}))) == 'invitation_request_expired'


def test_failed_receipt_commit_rolls_back_the_real_peer_reservation(gateway, monkeypatch):
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'http://127.0.0.1:9')
    params = dict(room_id='setup', home_install_id='install:home', authority_gateway_id='install:home',
        authority_epoch=1, member_id='peer', request_id='desktop-issuance-1', requested_at=time.time())
    first = asyncio.run(call(gateway.owner, 'groups.peer.invite', **params))
    assert isinstance(first, dict)
    db_path = _grant_db(gateway.adapter)
    with hosted_rooms._transaction(db_path, immediate=True) as conn:
        conn.execute("CREATE TRIGGER fail_setup_receipt BEFORE INSERT ON hosted_room_setup_invitations BEGIN SELECT RAISE(ABORT, 'fixture write failure'); END")
    rejected = asyncio.run(call(gateway.owner, 'groups.peer.invite', **(params | {
        'request_id': 'desktop-issuance-2', 'room_id': 'failed-setup'})))
    assert not isinstance(rejected, dict)
    assert not hosted_rooms.peer_room_is_reserved(db_path, room_id='failed-setup', target_profile='default')
    with hosted_rooms._transaction(db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM hosted_room_setup_invitations').fetchone()[0] == 1
