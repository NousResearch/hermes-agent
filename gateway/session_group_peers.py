"""Native operator-owned root target invitations and revocation receipts."""
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import uuid
from typing import Any

from gateway.session_contract import Principal

from hermes_state_runtime import RuntimeStoreError, _epoch

_PREFIX = 'gateway.peer.invite.v1.'


def invitation_lifetimes(params):
    ttl = params.get('ttl_seconds', 3600)
    status = params.get('status_ttl_seconds', ttl)
    if (type(ttl) not in (int, float) or type(status) not in (int, float)
            or not 60 <= ttl <= 86400 or not ttl <= status <= 2592000
            or not math.isfinite(ttl) or not math.isfinite(status)):
        raise ValueError('room grant lifetime is invalid')
    return float(ttl), float(status)


def invitation_preflight(params):
    """Use protocol limits AND the enforcing reservation store's limits."""
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import _DISPATCH_FIELDS, _identifier
    from gateway.platforms.api_server_room_grants import _ROOM_IDENTITY_FIELDS
    identity = {name: _DISPATCH_FIELDS[name](params.get(name), field=name)
                for name in _ROOM_IDENTITY_FIELDS}
    hosted_rooms._reservation_claims(dict(identity, target_profile='default'))
    if identity['authority_epoch'] > 2**63 - 1:
        raise ValueError('authority_epoch exceeds SQLite INTEGER range')
    if params.get('grant_id') is not None:
        _identifier(params['grant_id'], field='grant_id')
    return identity, invitation_lifetimes(params)


@dataclass(frozen=True)
class _NativeOperation:
    connection: Any
    authority: Any
    actor: Principal
    transport: object
    runner: object
    registry: object
    profile_id: str
    home: Path
    epoch: int
    instance_id: str
    db: Any
    paths: tuple

    def require_current(self, conn=None):
        from gateway.session_authorities import authority_for_home, served_profile_name
        from gateway.hosted_room_grant_state import grant_state_db_paths
        from hermes_constants import get_hermes_home
        connection, authority, actor = self.connection, self.authority, self.actor
        if (connection.authority is not authority or connection.actor is not actor
                or connection.native_owner is not True or connection.transport is not self.transport
                or not {'session:operator', 'session:control'} <= actor.capabilities
                or actor.profile_id != self.profile_id
                or authority.events.get(actor.transport_id) is not self.transport
                or getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) is not actor):
            raise RuntimeStoreError('permission_denied')
        if (authority.profile_id != self.profile_id or authority.runner is not self.runner
                or authority.db is not self.db or authority.instance_id != self.instance_id
                or authority.epoch != self.epoch
                or Path(get_hermes_home()).resolve() != self.home
                or served_profile_name(self.home) != 'default'
                or getattr(self.runner, 'session_authorities', None) is not self.registry
                or self.registry is None
                or authority_for_home(self.runner, self.home) is not authority
                or getattr(self.runner, 'session_authority', None) is not authority
                or Path(self.db.db_path).resolve() != self.home / 'state.db'
                or tuple(path.resolve() for path in grant_state_db_paths(self.home)) != self.paths
                or self.paths != (self.home / 'shared-state.db', self.home / 'state.db')):
            raise RuntimeStoreError('profile_mismatch')
        if conn is not None:
            if (not conn.in_transaction
                    or Path(conn.execute('PRAGMA database_list').fetchone()[2]).resolve() != self.paths[1]):
                raise RuntimeStoreError('profile_mismatch')
            _epoch(conn, self.epoch)
            row = conn.execute('SELECT instance_id FROM runtime_epoch WHERE singleton=1').fetchone()
            if row[0] != self.instance_id:
                raise RuntimeStoreError('stale_epoch')


def _native_owner(connection):
    from gateway.hosted_room_grant_state import grant_state_db_paths
    authority = connection.authority
    home = Path(authority.profile_id).resolve()
    # Freeze before any store read/lock can block. Never renew consent by reading
    # a successor's mutable epoch, actor or registry after that wait.
    operation = _NativeOperation(connection, authority, connection.actor, connection.transport,
        authority.runner, getattr(authority.runner, 'session_authorities', None),
        authority.profile_id, home, authority.epoch, authority.instance_id, authority.db,
        tuple(path.resolve() for path in grant_state_db_paths(home)))
    operation.require_current()
    return operation


def dispatch_group_peer(connection, method, params):
    operation = _native_owner(connection)
    authority = operation.authority
    if method != 'groups.peer.invite':
        return _revoke(operation, params, exact=method == 'groups.peer.revoke_exact')
    from gateway.config import Platform
    adapter = getattr(authority.runner, 'adapters', {}).get(Platform.API_SERVER)
    if adapter is None:
        raise RuntimeStoreError('canonical_room_peer_unsupported')
    return _invite(operation, adapter, params)


def _invite(operation, adapter, params):
    authority, actor = operation.authority, operation.actor
    run_store = adapter._run_idempotency_store
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import _identifier, issue_room_grant, decode_room_grant
    from gateway.platforms.api_server_room_grants import _local_room_catalog, _invitation_permissions
    from gateway.hosted_room_grant_state import reserve_grant_state
    from gateway.session_peer_target import target_policy, require_current_grant
    bound, paths, policy = target_policy(adapter)
    if bound is not authority or paths != operation.paths:
        raise RuntimeStoreError('profile_mismatch')
    request_id = _identifier(params.get('request_id'), field='request_id')
    identity, (ttl, status_ttl) = invitation_preflight(params)
    supplied_id = params.get('grant_id')
    installation = hosted_rooms.local_authority_gateway_id()
    _, catalog = _local_room_catalog(adapter, 'default', installation)
    if not catalog['text']:
        raise RuntimeStoreError('canonical_room_peer_unsupported')
    permissions = _invitation_permissions(adapter, 'default', catalog)
    endpoint = dict(catalog['endpoint'])
    intent = dict(identity, subject=actor.subject, home=operation.profile_id, epoch=operation.epoch,
                  endpoint=endpoint,
                  installation=installation, policy=policy, catalog=catalog, grant_id=supplied_id,
                  ttl_seconds=ttl, status_ttl_seconds=status_ttl, permissions=list(permissions))
    key = _PREFIX + hashlib.sha256(request_id.encode()).hexdigest()

    def require_target(conn):
        operation.require_current(conn)
        if adapter._run_idempotency_store is not run_store:
            raise RuntimeStoreError('profile_mismatch')
        current, current_paths, current_policy = target_policy(adapter, connection=conn)
        if current is not authority or current_paths != operation.paths or current_policy != policy:
            raise RuntimeStoreError('room_execution_policy_changed')
        _, current_catalog = _local_room_catalog(adapter, 'default', installation, _connection=conn)
        if (current_catalog != catalog
                or _invitation_permissions(adapter, 'default', current_catalog, _connection=conn) != permissions):
            raise RuntimeStoreError('room_capability_catalog_changed')
        operation.require_current(conn)

    def prepare(conn):
        require_target(conn)
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
                     execution_policy_digest=policy['policy_digest'], permissions=list(permissions),
                     issued_at=time.time(), ttl_seconds=ttl, status_ttl_seconds=status_ttl)
        receipt = dict(intent=intent, issue=issue, status='pending')
        operation.require_current(conn)
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, json.dumps(receipt)))
        return receipt, True

    receipt, created = operation.db._execute_write(prepare)
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
    with hosted_rooms._transaction(operation.paths[0], immediate=True) as shared:
        def complete(conn):
            require_target(conn)
            require_current_grant(shared, claims)
            require_current_grant(conn, claims)
            old = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
            if old != receipt:
                raise RuntimeStoreError('admission_conflict')
            operation.require_current(conn)
            if created:
                old['status'] = 'complete'
                old['token_sha256'] = claims['_token_sha256']
                conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(old), key))
        operation.db._execute_write(complete)
    operation.require_current()
    return dict(grant=token, target_profile='default', catalog=catalog, endpoint=endpoint,
                expires_at=claims['expires_at'], status_expires_at=claims['status_expires_at'])


def _revoke(operation, params, *, exact):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import gateway_room_grant_secret, decode_room_grant
    if set(params) != {'grant'}:
        raise RuntimeStoreError('invalid_params')
    claims = decode_room_grant(gateway_room_grant_secret(), params['grant'], permission='status',
                              allow_expired_for_revocation=True)
    if (claims['target_profile'] != 'default'
            or claims['target_install_id'] != hosted_rooms.local_authority_gateway_id()):
        raise RuntimeStoreError('permission_denied')
    revoke = hosted_rooms.revoke_room_grant_id if exact else hosted_rooms.revoke_room_grant_scope
    errors = []
    # Shared first, owner second. The owner transaction is the epoch fence even
    # for a shared-store write; an independent epoch read cannot authorize it.
    # Each store has a savepoint: one SQL failure must not undo a sibling deny.
    with ExitStack() as stack:
        shared = None
        try:
            shared = stack.enter_context(hosted_rooms._transaction(operation.paths[0], immediate=True))
        except Exception as exc:
            errors.append(exc)

        def write(owner):
            operation.require_current(owner)
            for path, conn in zip(operation.paths, (shared, owner)):
                if conn is None:
                    continue

                def authorize(actual):
                    operation.require_current(owner)
                    if (actual is not conn or not actual.in_transaction
                            or Path(actual.execute('PRAGMA database_list').fetchone()[2]).resolve() != path):
                        raise RuntimeStoreError('profile_mismatch')

                conn.execute('SAVEPOINT native_revoke')
                try:
                    revoke(path, claims=claims, expires_at=claims['status_expires_at'],
                           _connection=conn, _authorize_write=authorize)
                except Exception as exc:
                    conn.execute('ROLLBACK TO native_revoke')
                    errors.append(exc)
                finally:
                    conn.execute('RELEASE native_revoke')
        operation.db._execute_write(write)
    if errors:
        raise errors[0]
    operation.require_current()
    return {'revoked': True, 'exact': exact}
