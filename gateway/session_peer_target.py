"""Root RoomLink target binding and grant fences; no listener or executor."""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

from hermes_state_runtime import RuntimeStoreError, _epoch


def root_target(adapter, profile='default', *, connection=None):
    """Resolve a shared-listener target to its exact registered profile owner."""
    from gateway.config import Platform
    from gateway.session_authorities import authority_for_home, served_profile_name
    from gateway.hosted_room_grant_state import grant_state_db_paths
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home()).resolve()
    runner = adapter.gateway_runner
    authority = authority_for_home(runner, home)
    registry = getattr(runner, 'session_authorities', None)
    named = profile != 'default'
    if (served_profile_name(home) != profile
            or authority is None or authority.runner is not runner or registry is None
            or (named and (not getattr(runner.config, 'multiplex_profiles', False)
                           or registry.profile_name(authority) != profile
                           or getattr(runner, 'session_authority', None) is authority))
            or (not named and getattr(runner, 'session_authority', None) is not authority)
            or Path(authority.profile_id).resolve() != home
            or getattr(runner, 'adapters', {}).get(Platform.API_SERVER) is not adapter
            or adapter._ensure_session_db() is not authority.db
            or Path(authority.db.db_path).resolve() != home / 'state.db'):
        raise RuntimeStoreError('canonical_room_peer_unsupported')
    paths = tuple(path.resolve() for path in grant_state_db_paths(home))
    shared_home = home.parent.parent if named else home
    if paths != (shared_home / 'shared-state.db', home / 'state.db'):
        raise RuntimeStoreError('canonical_room_peer_unsupported')
    if not adapter._expected_api_key():
        raise RuntimeStoreError('canonical_room_peer_unsupported')
    from gateway.platforms.api_server_store import selected_run_idempotency_store
    store = selected_run_idempotency_store(adapter, home)
    if store is None:
        raise RuntimeStoreError('canonical_room_peer_unsupported')
    authority._require_admission_open()
    if connection is None:
        with authority.db.live_read_connection() as conn:
            if conn is None:
                raise RuntimeStoreError('canonical_room_peer_unsupported')
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


def authorize_dispatch(adapter, authority, shared, conn, token, dispatch, policy, *,
                       output_authorizer=None, output_evidence=None):
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
    from gateway.session_peer_route import catalog_matches_dispatch, require_room_route
    if dispatch.attachment_manifest_digest is not None:
        if 'attachment.stage' not in claims['permissions']:
            raise RuntimeStoreError('permission_denied')
        require_room_route(adapter, dispatch, connection=conn)
    if not catalog['text'] or not catalog_matches_dispatch(catalog, dispatch):
        raise RuntimeStoreError('room_capability_catalog_changed')
    require_current_grant(shared, claims)
    require_current_grant(conn, claims)
    _authorize_output_admission(adapter, authority, shared, conn, token, dispatch, policy,
                                claims, output_authorizer, output_evidence)


def _authorize_output_admission(adapter, authority, shared, conn, token, dispatch, policy,
                                claims, provider, evidence):
    """Concrete Output seam, called only by the protected NEW admission write.

    Output installs a synchronous adapter-bound ``_room_output_admission``.
    It must check its exact owner/evidence/current readiness on these already
    fenced connections, without acquiring a grant store, committing, or doing
    external work. Only literal True confirms consent; exceptions refuse NEW.
    The consent-capture provider must still be installed. Accepted replay never
    reaches this seam. No provider is needed for the old four/five-right grants.
    """
    rights = {'artifact.read', 'artifact.ack'} & set(claims['permissions'])
    if not rights:
        if evidence is not None:
            raise RuntimeStoreError('permission_denied')
        return
    if (rights != {'artifact.read', 'artifact.ack'}
            or not isinstance(provider, MethodType) or provider.__self__ is not adapter
            or getattr(adapter, '_room_output_admission', None) is not provider):
        raise RuntimeStoreError('room_output_unavailable')
    if (provider(authority, shared, conn, token, dispatch, policy, evidence) is not True
            or getattr(adapter, '_room_output_admission', None) is not provider):
        raise RuntimeStoreError('room_output_unavailable')
