"""Server-owned invite -> create -> register saga for local and peer members."""
import uuid
import sqlite3
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError


def _invite(home, member, room_id, gateway_id, db_path):
    from gateway.session_peers import _resolve, _credential
    from hermes_cli.subcommands import peer
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    alias, entry, profile = _resolve(home, member['route_id'])
    key = _credential(alias)
    if not key:
        raise RuntimeStoreError('peer_unauthenticated')
    # API credentials go only to the configured origin, never an advertised endpoint.
    client = PeerRunsHTTPClient(base_url=peer._base_url(entry, profile), api_key=key,
                                receipt_db_path=db_path)
    invitation = client.issue_invitation(
        room_id=room_id, home_install_id=gateway_id, authority_gateway_id=gateway_id,
        authority_epoch=1, member_id=member['member_id'], grant_id=f'grant-{uuid.uuid4().hex}')
    return client, invitation


def _peer_member(member, invitation, *, home):
    from gateway.hosted_room_peer import GatewayRoomCatalog, PROTOCOL_VERSION
    from gateway.session_peers import _resolve, _credential, _public_catalog
    from hermes_cli.subcommands.peer import DEFAULT_PROFILE_HANDLE
    alias, _, selected_profile = _resolve(home, member['route_id'])
    _public_catalog(invitation['catalog'], _credential(alias))
    catalog = GatewayRoomCatalog.from_mapping(invitation['catalog'])
    profile = invitation['target_profile']
    if ((selected_profile not in (None, DEFAULT_PROFILE_HANDLE) and profile != selected_profile)
            or profile != catalog.execution_policy.target_profile
            or PROTOCOL_VERSION not in catalog.protocol_versions or 'direct' not in catalog.link_modes):
        raise ValueError('incompatible RoomLink target')
    return dict(member_id=member['member_id'], handle=member['handle'],
                display_name=member.get('display_name', ''), profile=profile,
                target=dict(kind='peer', peer_id=member['route_id'], profile=profile,
                            installation_id=catalog.installation_id,
                            capability_digest=catalog.catalog_digest))


def _register(service, room, member, client, invitation):
    from gateway.hosted_room_peer import GatewayRoomCatalog
    from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    catalog = GatewayRoomCatalog.from_mapping(invitation['catalog'])
    # The persistent runtime receives a room-scoped credential only.
    scoped = PeerRunsHTTPClient(base_url=client.base_url, api_key='', receipt_db_path=service.db_path)
    probe = scoped.probe(grant=invitation['grant'])
    expected = dict(room_id=room['room_id'], member_id=member['member_id'],
                    home_install_id=room['authority_gateway_id'],
                    authority_gateway_id=room['authority_gateway_id'],
                    authority_epoch=room['authority_epoch'], target_profile=member['profile'])
    if (any(probe.get(k) != v for k, v in expected.items())
            or GatewayRoomCatalog.from_mapping(probe.get('catalog')) != catalog):
        raise ValueError('RoomLink scope or catalog changed')
    route = PeerMemberRoute(
        home_install_id=room['authority_gateway_id'], member_id=member['member_id'],
        target_install_id=catalog.installation_id, target_profile=member['profile'],
        capability_digest=catalog.catalog_digest,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        cancellation_scope_id=f"cancel-{room['room_id']}", trace_id=f'trace-{uuid.uuid4().hex}',
        grant=invitation['grant'])
    service.register_peer_route(room_id=room['room_id'], member_id=member['member_id'],
                                route=route, client=scoped, target_url=client.base_url, catalog=catalog)


def _rollback(connection, room_id, grants, created):
    from gateway.session_group_controls import _group
    from hermes_cli.config_backend import ConfigBackendUnavailable
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    from tui_gateway.hosted_room_service import _grant_revoke_is_terminal
    failures = False
    # Attempt every compensation even when another target is offline. Never include peer errors
    # in the wire refusal: they can carry credentials. A distinct reason makes cleanup retryable.
    for client, invitation in reversed(grants):
        grant = invitation.get('grant')
        if not isinstance(grant, str) or not grant:
            failures = True
            continue
        try:
            client.revoke_grant(grant=grant)
        except PeerRunsHTTPError as exc:
            failures |= not _grant_revoke_is_terminal(exc)
        except (OSError, ValueError, RuntimeError, sqlite3.Error):
            failures = True
    if created:
        try:
            _group(connection.authority, connection.actor, 'groups.disband', {'room_id': room_id})
        except (OSError, ValueError, RuntimeError, sqlite3.Error, ConfigBackendUnavailable):
            failures = True
    return failures


def create_mixed(connection, home, params):
    from gateway import hosted_rooms as rooms
    service = getattr(connection.authority, 'hosted_room_service', None)
    status = service.runtime.status() if service is not None else {}
    if not status.get('running') or status.get('stopping'):
        raise RuntimeStoreError('runtime_coordination_required')
    if Path(service.db_path).resolve() != Path(connection.authority.db.db_path).resolve():
        raise RuntimeStoreError('profile_mismatch')
    # Serialize with driver policy and other mixed creates. No input can start work until all
    # grants have been probed and persisted. Existing rooms are never rollback candidates.
    with service._policy_lock:
        return _create(connection, home, params, service, rooms)


def _create(connection, home, params, service, rooms):
    from gateway.session_group_controls import _group
    from gateway.hosted_room_discussion import validate_roster
    from hermes_cli.config_backend import ConfigBackendUnavailable
    room_id, members = params['room_id'], params['members']
    gateway_id = rooms.local_authority_gateway_id()
    try:
        rooms.room_state(service.db_path, room_id=room_id, include_disbanded=True)
    except rooms.RoomHistoryExpiredError:
        raise RuntimeStoreError('room_already_exists')
    except rooms.RoomNotFoundError:
        pass
    else:
        raise RuntimeStoreError('room_already_exists')
    grants, normalized, peers, created = [], [], [], False
    try:
        for member in members:
            if member['kind'] == 'local':
                normalized.append({k: v for k, v in member.items() if k != 'kind'})
            else:
                client, invitation = _invite(home, member, room_id, gateway_id, service.db_path)
                grants.append((client, invitation))
                row = _peer_member(member, invitation, home=home)
                normalized.append(row)
                peers.append((row, client, invitation))
        # Same validation and ownership transaction as ordinary groups.create.
        from gateway.session_hosted_members import member_roster, refuse_unseatable
        homes, refused = member_roster(connection.authority, siblings=connection.host_scoped)
        refuse_unseatable(normalized, refused)
        validate_roster(normalized, local_profiles=homes)
        result = _group(connection.authority, connection.actor, 'groups.create',
                        dict(room_id=room_id, name=params['name'], members=normalized),
                        host_scope=connection.host_scoped)
        created = not result['room'].get('idempotent', False)
        if not created:
            raise RuntimeStoreError('room_already_exists')
        for member, client, invitation in peers:
            _register(service, result['room'], member, client, invitation)
        return result
    except (Exception, ConfigBackendUnavailable) as exc:
        failed = _rollback(connection, room_id, grants, created)
        # A lost invitation response may have minted a grant whose token we never received.
        # Do not claim complete compensation when the target cannot prove non-admission.
        failed |= bool(getattr(exc, 'ambiguous', False))
        reason = 'mixed_room_rollback_incomplete' if failed else 'mixed_room_setup_failed'
        raise RuntimeStoreError(reason) from exc
