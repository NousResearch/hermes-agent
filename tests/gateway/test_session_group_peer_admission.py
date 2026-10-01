"""Participant permission is checked in the accepting canonical writer transaction."""
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret, room_grant_token_digest
from gateway.platforms.api_server_room_grants import authorize_room_admission
from hermes_state_runtime import RuntimeStoreError, admit_session_input
from tests.gateway.test_session_group_peers import gateway, invite, linked_room  # noqa: F401
from gateway.session_group_peers import room_link


@pytest.mark.asyncio
@pytest.mark.parametrize('retirement', ['exact', 'scope', 'reservation'])
async def test_revoke_between_request_check_and_accepting_write_refuses_new_admission(
        gateway, monkeypatch, retirement):
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'http://127.0.0.1:9999')
    gateway.adapter.gateway_runner = SimpleNamespace(session_authority=gateway.authority)
    room = await linked_room(gateway, room_link(gateway.authority)['catalog'])
    grant = (await invite(gateway, room))['grant']
    request = SimpleNamespace(headers={'Authorization': 'HermesRoom ' + grant})
    guard = authorize_room_admission(gateway.adapter, request)
    claims = decode_room_grant(gateway_room_grant_secret(), grant, permission='dispatch')
    gateway.db.create_session(session_id='peer-admission', source='api')
    args = dict(epoch=gateway.authority.epoch, principal_id='api', session_id='peer-admission',
                request_id='accepted', payload={'text': 'same input'}, _authorize_write=guard)
    accepted = admit_session_input(gateway.db, **args)
    if retirement == 'exact':
        hosted_rooms.revoke_room_grant_token(gateway.db.db_path, claims=claims,
            token_sha256=room_grant_token_digest(grant), expires_at=claims['status_expires_at'])
    elif retirement == 'scope':
        hosted_rooms.revoke_room_grant_scope(gateway.db.db_path, claims=claims,
                                           expires_at=claims['status_expires_at'])
    else:
        gateway.db._execute_write(lambda conn: conn.execute(
            'UPDATE hosted_room_peer_reservations SET authority_epoch=authority_epoch+1'))
    # An exact replay observes its existing receipt even after the bearer is retired.
    assert admit_session_input(gateway.db, **args)['admission_id'] == accepted['admission_id']
    with pytest.raises(RuntimeStoreError, match='room_reauthorization_required'):
        admit_session_input(gateway.db, **{**args, 'request_id': 'new'})
    with gateway.db._read_ctx() as conn:
        assert conn.execute('SELECT request_id FROM session_admissions').fetchall()[0][0] == 'accepted'
        assert conn.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0] == 1


@pytest.mark.asyncio
async def test_api_turn_forwards_participant_fence_to_accepting_writer(gateway, monkeypatch):
    from gateway import session_api_turn
    from gateway.session_contract import SessionRef
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'http://127.0.0.1:9999')
    gateway.authority._require_admission_open = lambda: None
    gateway.adapter.gateway_runner = SimpleNamespace(session_authority=gateway.authority)
    monkeypatch.setattr(gateway.adapter, '_ensure_session_db', lambda: gateway.db)
    room = await linked_room(gateway, room_link(gateway.authority)['catalog'])
    token = (await invite(gateway, room))['grant']
    claims = decode_room_grant(gateway_room_grant_secret(), token, permission='status')
    guard = authorize_room_admission(gateway.adapter, SimpleNamespace(headers={'Authorization': 'HermesRoom ' + token}))
    gateway.db.create_session(session_id='writer-fence', source='api')
    monkeypatch.setattr(session_api_turn, 'bind_api_session', lambda authority, sid, **kwargs: SessionRef(authority.profile_id, sid))
    def revoke_after_preparation(*args):
        hosted_rooms.revoke_room_grant_scope(gateway.db.db_path, claims=claims, expires_at=claims['status_expires_at'])
    monkeypatch.setattr(session_api_turn, 'check_api_turn', revoke_after_preparation)
    with pytest.raises(RuntimeStoreError, match='room_reauthorization_required'):
        session_api_turn.admit_api_turn(gateway.adapter, session_id='writer-fence', request_id='race',
            user_message='must not execute', conversation_history=[], _authorize_write=guard)
    with gateway.db._read_ctx() as conn:
        assert conn.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0] == 0
