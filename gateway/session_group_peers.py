"""Native operator-owned root target invitations and revocation receipts."""
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from hermes_state_runtime import RuntimeStoreError, _epoch

_PREFIX = 'gateway.peer.invite.v1.'
_PERMISSIONS = ('approve', 'dispatch', 'status', 'stop')


def invitation_lifetimes(params):
    ttl = params.get('ttl_seconds', 3600)
    status = params.get('status_ttl_seconds', ttl)
    if (type(ttl) not in (int, float) or type(status) not in (int, float)
            or not math.isfinite(ttl) or not math.isfinite(status)
            or not 60 <= ttl <= 86400 or not ttl <= status <= 2592000):
        raise ValueError('room grant lifetime is invalid')
    return float(ttl), float(status)


def _native_owner(connection):
    from gateway.session_authorities import authority_for_home, served_profile_name
    authority, actor = connection.authority, connection.actor
    home = Path(authority.profile_id).resolve()
    if (connection.native_owner is not True
            or not {'session:operator', 'session:control'} <= actor.capabilities
            or actor.profile_id != authority.profile_id
            or actor.transport_id not in authority.events):
        raise RuntimeStoreError('permission_denied')
    if (served_profile_name(home) != 'default'
            or getattr(authority.runner, 'session_authorities', None) is None
            or Path(authority.db.db_path).resolve() != home / 'state.db'
            or authority_for_home(authority.runner, home) is not authority):
        raise RuntimeStoreError('profile_mismatch')
    with authority.db._read_ctx() as conn:
        _epoch(conn, authority.epoch)
    return authority


def dispatch_group_peer(connection, method, params):
    authority = _native_owner(connection)
    if method != 'groups.peer.invite':
        return _revoke(authority, params, exact=method == 'groups.peer.revoke_exact')
    from gateway.config import Platform
    adapter = getattr(authority.runner, 'adapters', {}).get(Platform.API_SERVER)
    if adapter is None:
        raise RuntimeStoreError('canonical_room_peer_unsupported')
    return _invite(authority, connection.actor, adapter, params)


def _invite(authority, actor, adapter, params):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import _DISPATCH_FIELDS, _identifier, issue_room_grant, decode_room_grant
    from gateway.platforms.api_server_room_grants import _ROOM_IDENTITY_FIELDS, _local_room_catalog
    from gateway.hosted_room_grant_state import reserve_grant_state
    from gateway.session_peer_target import target_policy, grant_fence, require_current_grant
    bound, paths, policy = target_policy(adapter)
    if bound is not authority:
        raise RuntimeStoreError('profile_mismatch')
    request_id = _identifier(params.get('request_id'), field='request_id')
    identity = {name: _DISPATCH_FIELDS[name](params.get(name), field=name) for name in _ROOM_IDENTITY_FIELDS}
    ttl, status_ttl = invitation_lifetimes(params)
    supplied_id = params.get('grant_id')
    if supplied_id is not None:
        _identifier(supplied_id, field='grant_id')
    installation = hosted_rooms.local_authority_gateway_id()
    _, catalog = _local_room_catalog(adapter, 'default', installation)
    if not catalog['text']:
        raise RuntimeStoreError('canonical_room_peer_unsupported')
    host = adapter._host
    endpoint = f'http://{host if ":" not in host else "[" + host + "]"}:{adapter._port}'
    intent = dict(identity, subject=actor.subject, home=authority.profile_id, epoch=authority.epoch,
                  endpoint=endpoint,
                  installation=installation, policy=policy, catalog=catalog, grant_id=supplied_id,
                  ttl_seconds=ttl, status_ttl_seconds=status_ttl, permissions=list(_PERMISSIONS))
    key = _PREFIX + hashlib.sha256(request_id.encode()).hexdigest()

    def prepare(conn):
        _epoch(conn, authority.epoch)
        old = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        if old:
            receipt = json.loads(old[0])
            if receipt['intent'] != intent:
                raise RuntimeStoreError('admission_conflict')
            if receipt['status'] != 'complete':
                raise RuntimeStoreError('room_invitation_pending')
            return receipt, False
        issue = dict(identity, grant_id=supplied_id or 'grant-' + uuid.uuid4().hex,
                     target_install_id=installation, target_profile='default',
                     execution_policy_digest=policy['policy_digest'], permissions=list(_PERMISSIONS),
                     issued_at=time.time(), ttl_seconds=ttl, status_ttl_seconds=status_ttl)
        receipt = dict(intent=intent, issue=issue, status='pending')
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, json.dumps(receipt)))
        return receipt, True

    receipt, created = authority.db._execute_write(prepare)
    # Intent is durable before signing/reservation. A pending retry never issues
    # another token or re-reserves (which could undo a concurrent scope revoke).
    token = issue_room_grant(adapter._room_grant_secret(), **receipt['issue'])
    claims = decode_room_grant(adapter._room_grant_secret(), token, permission='status')
    if not created and receipt.get('token_sha256') != claims['_token_sha256']:
        raise RuntimeStoreError('room_reauthorization_required')
    if created:
        # Existing compensation preserves partial-deny/CAS semantics. A crash or
        # failure leaves the request pending, not a mint-on-retry capability.
        reserve_grant_state(paths, claims=claims, expires_at=claims['status_expires_at'])
    with grant_fence(adapter) as (bound, shared):
        def complete(conn):
            current, _, current_policy = target_policy(adapter, connection=conn)
            if current is not authority or current_policy != policy:
                raise RuntimeStoreError('room_execution_policy_changed')
            require_current_grant(shared, claims)
            require_current_grant(conn, claims)
            old = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
            if old != receipt:
                raise RuntimeStoreError('admission_conflict')
            if created:
                old['status'] = 'complete'
                old['token_sha256'] = claims['_token_sha256']
                conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(old), key))
        authority.db._execute_write(complete)
    return dict(grant=token, target_profile='default', catalog=catalog, endpoint=endpoint,
                expires_at=claims['expires_at'], status_expires_at=claims['status_expires_at'])


def _revoke(authority, params, *, exact):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import gateway_room_grant_secret, decode_room_grant
    from gateway.hosted_room_grant_state import grant_state_db_paths, revoke_grant_state
    if set(params) != {'grant'}:
        raise RuntimeStoreError('invalid_params')
    claims = decode_room_grant(gateway_room_grant_secret(), params['grant'], permission='status',
                              allow_expired_for_revocation=True)
    if (claims['target_profile'] != 'default'
            or claims['target_install_id'] != hosted_rooms.local_authority_gateway_id()):
        raise RuntimeStoreError('permission_denied')
    revoke_grant_state(grant_state_db_paths(authority.profile_id), claims=claims,
                       expires_at=claims['status_expires_at'], exact=exact)
    return {'revoked': True, 'exact': exact}
