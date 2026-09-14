"""Root RoomLink target binding and grant fences; no listener or executor."""
from contextlib import contextmanager
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError, _epoch


def root_target(adapter, profile='default', *, connection=None):
    """Resolve the configured shared listener to its registered owning root DB."""
    from gateway.config import Platform
    from gateway.session_authorities import authority_for_home, served_profile_name
    from gateway.hosted_room_grant_state import grant_state_db_paths
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home()).resolve()
    runner = adapter.gateway_runner
    authority = authority_for_home(runner, home)
    if (profile != 'default' or served_profile_name(home) != 'default'
            or authority is None or authority.runner is not runner
            or getattr(runner, 'session_authority', None) is not authority
            or getattr(runner, 'session_authorities', None) is None
            or Path(authority.profile_id).resolve() != home
            or getattr(runner, 'adapters', {}).get(Platform.API_SERVER) is not adapter
            or adapter._ensure_session_db() is not authority.db
            or Path(authority.db.db_path).resolve() != home / 'state.db'):
        raise RuntimeStoreError('canonical_room_peer_unsupported')
    paths = tuple(path.resolve() for path in grant_state_db_paths(home))
    if paths != (home / 'shared-state.db', home / 'state.db'):
        raise RuntimeStoreError('canonical_room_peer_unsupported')
    store = getattr(adapter, '_run_idempotency_store', None)
    if (store is None or not store.durable
            or Path(store._db_path).resolve() != home / 'runs_idempotency.db'
            or not adapter._expected_api_key()):
        raise RuntimeStoreError('canonical_room_peer_unsupported')
    actual = store._conn.execute('PRAGMA database_list').fetchone()[2]
    if Path(actual).resolve() != Path(store._db_path).resolve():
        raise RuntimeStoreError('canonical_room_peer_unsupported')
    authority._require_admission_open()
    if connection is None:
        with authority.db._read_ctx() as conn:
            _epoch(conn, authority.epoch)
    else:
        _epoch(connection, authority.epoch)
    return authority, paths


def target_policy(adapter, profile='default', *, connection=None):
    from gateway.hosted_room_execution_policy import execution_policy_mapping
    from gateway.session_authorities import owner_scope
    authority, paths = root_target(adapter, profile, connection=connection)
    with owner_scope(authority):
        policy = execution_policy_mapping(target_profile=profile)
    if policy['approval_mode'] == 'off':
        raise RuntimeStoreError('room_execution_policy_changed')
    return authority, paths, policy


def require_current_grant(conn, claims):
    from gateway.hosted_rooms import room_grant_is_revoked_on_conn, peer_room_grant_is_current_on_conn
    if (room_grant_is_revoked_on_conn(conn, claims=claims)
            or not peer_room_grant_is_current_on_conn(conn, claims=claims)):
        raise RuntimeStoreError('room_reauthorization_required')


@contextmanager
def grant_fence(adapter, profile='default'):
    """Shared first, owning SessionDB second (by the caller), through commit.

    This is serialization, NOT crash-atomicity across two WAL databases. Never
    reacquire either grant DB from a predicate inside the owning transaction.
    """
    from gateway import hosted_rooms
    authority, paths = root_target(adapter, profile)
    with hosted_rooms._transaction(paths[0], immediate=True) as shared:
        yield authority, shared


def authorize_dispatch(adapter, authority, shared, conn, token, dispatch, policy):
    from gateway.hosted_room_peer import verify_room_grant
    from gateway.platforms.api_server_room_grants import _local_room_catalog
    current, _, actual = target_policy(adapter, dispatch.target_profile, connection=conn)
    if current is not authority or actual != policy:
        raise RuntimeStoreError('room_execution_policy_changed')
    from gateway import hosted_rooms
    if dispatch.target_install_id != hosted_rooms.local_authority_gateway_id():
        raise RuntimeStoreError('permission_denied')
    claims = verify_room_grant(adapter._room_grant_secret(), token, dispatch, permission='dispatch')
    # Policy has already been derived on the transaction's owning connection.
    _, catalog = _local_room_catalog(adapter, dispatch.target_profile, dispatch.target_install_id,
                                     _connection=conn)
    if not catalog['text'] or catalog['catalog_digest'] != dispatch.capability_digest:
        raise RuntimeStoreError('room_capability_catalog_changed')
    require_current_grant(shared, claims)
    require_current_grant(conn, claims)
