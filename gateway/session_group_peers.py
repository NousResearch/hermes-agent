"""Cross-gateway members of gateway-owned Group Chats: text-only, default-profile targets.

The target gateway's operator mints a room-scoped grant (``groups.peer.invite``) or revokes that
scope (``groups.peer.revoke``). The room's home binds the grant to the room's pinned peer member
(``groups.peer.register``). Dispatch, Stop, recovery and Disband then run on the canonical hosted
service like any other member (``CanonicalHostedRoomService`` inherits the peer transport).
"""
import logging
import sqlite3
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

TARGET_METHODS = frozenset({'groups.peer.invite', 'groups.peer.revoke'})
_GRANT_ERRORS = frozenset({'invalid_room_grant', 'room_reauthorization_required'})


def _api_server(authority):
    from gateway.config import Platform
    adapters = getattr(getattr(authority, 'runner', None), 'adapters', None) or {}
    return adapters.get(Platform.API_SERVER)


def room_link(authority):
    """Whether this gateway can host a member of another gateway's room; else the first missing piece."""
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import gateway_room_grant_secret
    from gateway.platforms.api_server_room_grants import _local_room_catalog
    from gateway.session_authorities import served_profile_name
    if served_profile_name(Path(authority.profile_id)) != 'default':
        return {'enabled': False, 'reason': 'default_profile_required'}
    adapter = _api_server(authority)
    if adapter is None:
        return {'enabled': False, 'reason': 'api_server_required'}
    if not getattr(getattr(adapter, '_run_idempotency_store', None), 'durable', False):
        return {'enabled': False, 'reason': 'durable_run_storage_required'}
    try:
        gateway_room_grant_secret()
    except (OSError, RuntimeError, ValueError):
        return {'enabled': False, 'reason': 'gateway_roomlink_secret_unavailable'}
    try:
        _, catalog = _local_room_catalog(adapter, 'default', hosted_rooms.local_authority_gateway_id())
    except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError):
        # For example approvals off: a remote turn must never run without approval prompts.
        return {'enabled': False, 'reason': 'execution_policy_unsupported'}
    if not catalog['endpoint'].get('available'):
        return {'enabled': False, 'reason': 'endpoint_required'}
    return {'enabled': True, 'profile': 'default', 'catalog': catalog, 'endpoint': catalog['endpoint']}


def dispatch_target(authority, method, params):
    """Operator-only target controls; ``room_id`` names the home's room, not a local one."""
    return (_invite if method == 'groups.peer.invite' else _revoke)(authority, params)


def _invite(authority, params):
    from gateway.platforms.api_server_room_grants import _issue_invitation
    identity = ('room_id', 'home_install_id', 'authority_gateway_id', 'member_id')
    if (not all(isinstance(params.get(k), str) and params[k] for k in identity)
            or type(params.get('authority_epoch')) is not int or not 1 <= params['authority_epoch'] < 2**63
            or type(params.get('ttl_seconds', 3600)) not in (int, float)):
        raise RuntimeStoreError('invalid_params')
    if not room_link(authority)['enabled']:
        raise RuntimeStoreError('room_link_unavailable')
    invitation = _issue_invitation(_api_server(authority), params, 'default')
    return {'grant': invitation['grant'], 'target_profile': invitation['target_profile'],
            'catalog': invitation['catalog'], 'endpoint': invitation['catalog']['endpoint']}


def _revoke(authority, params):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import HostedRoomPeerError, decode_room_grant, gateway_room_grant_secret
    from gateway.session_authorities import served_profile_name
    if not isinstance(params.get('grant'), str) or not params['grant']:
        raise RuntimeStoreError('invalid_params')
    try:
        claims = decode_room_grant(gateway_room_grant_secret(), params['grant'], permission='status')
    except HostedRoomPeerError as exc:
        raise RuntimeStoreError('invalid_room_grant') from exc
    if (claims['target_profile'], claims['target_install_id']) != (
            served_profile_name(Path(authority.profile_id)), hosted_rooms.local_authority_gateway_id()):
        raise RuntimeStoreError('permission_denied')
    hosted_rooms.revoke_room_grant_scope(
        hosted_rooms.default_db_path(), claims=claims,
        expires_at=float(claims.get('status_expires_at', claims['expires_at'])))
    return {'revoked': True}


def _probe(client, grant, catalog, scope):
    """Check a grant on the member's gateway: reachable, the same catalog, the same room scope."""
    from gateway.hosted_room_peer import GatewayRoomCatalog, HostedRoomPeerError
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    try:
        probe = client.probe(grant=grant)
        live = GatewayRoomCatalog.from_mapping(probe.get('catalog'))
    except PeerRunsHTTPError as exc:
        raise RuntimeStoreError(exc.error_code if exc.error_code in _GRANT_ERRORS else 'peer_unreachable') from exc
    except HostedRoomPeerError as exc:
        raise RuntimeStoreError('peer_target_mismatch') from exc
    if live != catalog or any(probe.get(k) != v for k, v in scope.items()):
        raise RuntimeStoreError('peer_target_mismatch')


def register(service, params):
    """Bind a target grant to the room's pinned peer member, after a live scoped probe."""
    from gateway.hosted_room_peer import (
        GatewayRoomCatalog, HostedRoomPeerError, PROTOCOL_VERSION, validate_room_link_url)
    from gateway.hosted_rooms import HostedRoomError
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    from tui_gateway.hosted_room_peer_transport import PeerMemberRoute, route_identity
    room_id, member_id, profile, grant = (params.get(k) for k in ('room_id', 'member_id', 'target_profile', 'grant'))
    if not all(isinstance(v, str) and v for v in (room_id, member_id, profile, grant)):
        raise RuntimeStoreError('invalid_params')
    try:
        target_url, security = validate_room_link_url(params.get('target_url'))
        catalog = GatewayRoomCatalog.from_mapping(params.get('catalog'))
    except (HostedRoomPeerError, TypeError, KeyError) as exc:
        raise RuntimeStoreError('invalid_params') from exc
    gateway_id, epoch = service._owned_authority(room_id)
    member = next((m for m in service._room(room_id)['members'] if m['member_id'] == member_id), {})
    target = member.get('target') or {}
    if (target.get('kind') != 'peer' or target.get('installation_id') != catalog.installation_id
            or target.get('capability_digest') != catalog.catalog_digest
            or target.get('profile') != profile or profile != catalog.execution_policy.target_profile):
        raise RuntimeStoreError('peer_target_mismatch')
    if (profile != 'default' or not catalog.text or catalog.attachments
            or PROTOCOL_VERSION not in catalog.protocol_versions or 'direct' not in catalog.link_modes):
        raise RuntimeStoreError('peer_target_unsupported')
    if target_url != catalog.endpoint_url:
        # The pinned digest covers the endpoint the member's gateway advertises: the grant goes
        # there and nowhere else, so another host can neither read it nor answer for that gateway.
        raise RuntimeStoreError('peer_target_mismatch')
    client = PeerRunsHTTPClient(base_url=target_url, api_key='', receipt_db_path=service.db_path)
    # The network probe holds no lock, so Stop and Disband are never held behind a slow peer.
    _probe(client, grant, catalog, {
        'room_id': room_id, 'home_install_id': gateway_id, 'authority_gateway_id': gateway_id,
        'authority_epoch': epoch, 'member_id': member_id, 'target_profile': profile})
    # The home derives the route's identity (a caller cannot choose it), so a re-registered grant
    # replays an accepted dispatch byte for byte instead of conflicting with it.
    cancellation_scope_id, trace_id = route_identity(
        db_path=service.db_path, home_install_id=gateway_id, room_id=room_id, member_id=member_id)
    route = PeerMemberRoute(
        home_install_id=gateway_id, member_id=member_id, target_install_id=catalog.installation_id,
        target_profile=profile, capability_digest=catalog.catalog_digest,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        cancellation_scope_id=cancellation_scope_id, trace_id=trace_id, grant=grant)
    with service.peer_route_lock:
        # The probe took time: publish only while the room is still the one it was checked against.
        try:
            refused = (None if service._owned_authority(room_id) == (gateway_id, epoch)
                       else RuntimeStoreError('peer_target_mismatch'))
        except HostedRoomError as exc:  # disbanded meanwhile
            refused = exc
        if refused is None:
            service.register_peer_route(room_id=room_id, member_id=member_id, route=route, client=client,
                                        target_url=target_url, catalog=catalog)
    if refused is not None:
        # The member's gateway has seen this grant for a room that ended or moved: revoke it there
        # (outside the lock), so Disband leaves no grant this home was handed.
        _revoke_quietly(client, grant)
        raise refused
    return {'registered': True, 'mode': 'direct', 'transport_security': security,
            'target_install_id': catalog.installation_id, 'target_profile': profile}


def _revoke_quietly(client, grant):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    try:
        client.revoke_grant(grant=grant)
    except PeerRunsHTTPError as exc:
        logger.warning('peer grant for an ended room could not be revoked; it expires on its own: %s', exc)
