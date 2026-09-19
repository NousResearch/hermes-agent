"""Read-only canonical API retries before binding or preparing input custody."""
import copy
import json

from hermes_state_runtime import RuntimeStoreError, _epoch, _row, admission_fingerprint


def authenticate_room_retry(adapter, authority, session_id, dispatch, token):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import verify_room_grant
    from gateway.platforms.api_server_room_grants import _require_current_room_grant
    from gateway.session_api import hosted_session_id
    from gateway.session_peer_target import root_target
    try:
        claims = verify_room_grant(adapter._room_grant_secret(), token, dispatch, permission='dispatch')
        verify_room_grant(adapter._room_grant_secret(), token, dispatch, permission='status')
        owner, _ = root_target(adapter, dispatch.target_profile)
        if owner is not authority or dispatch.target_install_id != hosted_rooms.local_authority_gateway_id():
            raise RuntimeStoreError('permission_denied')
        if session_id != hosted_session_id(dispatch):
            raise RuntimeStoreError('admission_conflict')
        _require_current_room_grant(adapter, claims)
    except RuntimeStoreError:
        raise
    except ValueError as exc:
        raise RuntimeStoreError('permission_denied') from exc


def replay_room_admission(authority, *, session_id, request_id, payload):
    from hermes_state_terminal import identity_key, terminal_admission
    with authority.db._read_ctx() as conn:
        _epoch(conn, authority.epoch)
        old = conn.execute("SELECT * FROM session_admissions WHERE principal_id='api' "
                           'AND target_session_id=? AND request_id=?', (session_id, request_id)).fetchone()
        if old is None:
            saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                 (identity_key('api', session_id, request_id),)).fetchone()
            old = terminal_admission(conn, json.loads(saved[0])) if saved else None
            if saved and old is None:
                raise RuntimeStoreError('storage_unavailable')
        if old is None:
            return None
        frozen = json.loads(old['payload_json'])
        turn = frozen.get('api_turn_v1')
        if not isinstance(turn, dict) or not isinstance(turn.get('settings'), dict):
            # No HTTP fingerprint at this internal entry. An erased payload
            # cannot be reconstructed, and must never reach Files preparation.
            raise RuntimeStoreError('storage_unavailable')
        digest = admission_fingerprint(canonical_target=session_id, payload={'input': frozen, 'intent': 'queue'})
        if digest != old['payload_digest']:
            raise RuntimeStoreError('admission_conflict')
        candidate = copy.deepcopy(frozen)
        candidate['api_turn_v1']['settings'].pop('room_input_media', None)
        if candidate != payload:
            raise RuntimeStoreError('admission_conflict')
        return _row(old)
