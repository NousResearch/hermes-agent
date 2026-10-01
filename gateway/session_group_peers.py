"""Canonical cross-gateway Group Chat members: text-only, default-profile targets.

The target gateway's operator mints a room-scoped grant (``groups.peer.invite``) or
revokes that scope (``groups.peer.revoke``). The room's home binds the grant to the
room's pinned peer member (``groups.peer.register``). Dispatch, Stop, recovery and
Disband then run on the canonical hosted service like any other member.
"""
import hashlib
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

TARGET_METHODS = frozenset({'groups.peer.invite', 'groups.peer.revoke'})
_GRANT_ERRORS = frozenset({'invalid_room_grant', 'room_reauthorization_required'})


def _api_server(authority):
    from gateway.config import Platform
    adapters = getattr(getattr(authority, 'runner', None), 'adapters', None) or {}
    return adapters.get(Platform.API_SERVER)


def room_link(authority):
    """Whether this gateway can host a peer member, honestly: else the first missing piece."""
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
    except Exception:
        return {'enabled': False, 'reason': 'gateway_roomlink_secret_unavailable'}
    try:
        _, catalog = _local_room_catalog(adapter, 'default', hosted_rooms.local_authority_gateway_id())
    except Exception:
        # For example YOLO approvals: a remote turn must never run without approval prompts.
        return {'enabled': False, 'reason': 'execution_policy_unsupported'}
    if not catalog['endpoint'].get('available'):
        return {'enabled': False, 'reason': 'endpoint_required'}
    return {'enabled': True, 'profile': 'default', 'catalog': catalog, 'endpoint': catalog['endpoint'],
            'authentication': 'proof-v1'}


def dispatch_target(authority, method, params):
    """Operator-only target controls; ``room_id`` names the home's room, not a local one."""
    if method == 'groups.peer.invite':
        return _invite(authority, params)
    return _revoke(authority, params)


def _invite(authority, params):
    from gateway.platforms.api_server_room_grants import _issue_invitation
    identity = ('room_id', 'home_install_id', 'authority_gateway_id', 'member_id')
    if (not all(isinstance(params.get(k), str) and params[k] for k in identity)
            or type(params.get('authority_epoch')) is not int or not 1 <= params['authority_epoch'] < 2**63
            or any(type(params.get(k, 3600)) not in (int, float) for k in ('ttl_seconds', 'status_ttl_seconds'))):
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
    from gateway.platforms.api_server_room_grants import _grant_db
    hosted_rooms.revoke_room_grant_scope(
        _grant_db(_api_server(authority)), claims=claims,
        expires_at=float(claims.get('status_expires_at', claims['expires_at'])))
    return {'revoked': True}


def probe_route(client, grant, catalog, scope):
    """Check a grant on the member's gateway: reachable, the same catalog, the same room scope."""
    from gateway.hosted_room_peer import GatewayRoomCatalog, HostedRoomPeerError
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    try:
        probe = client.probe(grant=grant)
        live = GatewayRoomCatalog.from_mapping(probe.get('catalog'))
    except PeerRunsHTTPError as exc:
        reason = exc.error_code if exc.error_code in _GRANT_ERRORS else 'peer_unreachable'
        raise RuntimeStoreError(reason) from exc
    except HostedRoomPeerError as exc:
        raise RuntimeStoreError('peer_target_mismatch') from exc
    if live != catalog or any(probe.get(k) != v for k, v in scope.items()):
        raise RuntimeStoreError('peer_target_mismatch')


def register(service, params):
    """Bind a target grant to the room's pinned peer member, after a live scoped probe."""
    from gateway.hosted_room_peer import (
        GatewayRoomCatalog, HostedRoomPeerError, PROTOCOL_VERSION, validate_room_link_url)
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
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
    client = PeerRunsHTTPClient(base_url=target_url, api_key='', receipt_db_path=service.db_path,
                                proof_install_id=catalog.installation_id)
    # Route identity is derived, never random: a re-registered grant must replay an accepted
    # dispatch byte for byte, or the target's idempotency check reads the replay as a new run.
    seed = '\0'.join((gateway_id, room_id, member_id)).encode()
    route = PeerMemberRoute(
        home_install_id=gateway_id, member_id=member_id, target_install_id=catalog.installation_id,
        target_profile=profile, capability_digest=catalog.catalog_digest,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        cancellation_scope_id='cancel-' + room_id,
        trace_id='trace-' + hashlib.sha256(seed).hexdigest()[:32], grant=grant)
    from gateway import hosted_room_links as links
    from gateway import session_group_peer_cleanup as cleanup
    cleanup_key = None
    with service.peer_route_lock:
        previous = links.load_room_link(service.db_path, room_id=room_id, member_id=member_id)
        if previous is None or previous.grant != grant:
            cleanup_key = cleanup.retain(service.db_path, links.make_stored_link(
                room_id=room_id, member_id=member_id, target_url=target_url, target_profile=profile,
                grant=grant, catalog=catalog, cancellation_scope_id=route.cancellation_scope_id,
                trace_id=route.trace_id))
            service._peer_cleanup_inflight.add(cleanup_key)
    try:
        # The network probe holds neither publication nor policy lock: Stop/Disband can proceed.
        probe_route(client, grant, catalog, {
            'room_id': room_id, 'home_install_id': gateway_id, 'authority_gateway_id': gateway_id,
            'authority_epoch': epoch, 'member_id': member_id, 'target_profile': profile})
        with service.peer_route_lock:
            if service._owned_authority(room_id) != (gateway_id, epoch):
                raise RuntimeStoreError('peer_target_mismatch')
            service.register_peer_route(room_id=room_id, member_id=member_id, route=route, client=client,
                                        target_url=target_url, catalog=catalog)
    finally:
        with service.peer_route_lock:
            service._peer_cleanup_inflight.discard(cleanup_key)
    return {'registered': True, 'mode': 'direct', 'transport_security': security,
            'target_install_id': catalog.installation_id, 'target_profile': profile}


class _RefusedTurn:
    """A peer turn that fails before anything is dispatched instead of running unobserved."""

    def __init__(self, reason):
        self.reason = reason

    def _refuse(self, **_):
        raise RuntimeError(self.reason)

    resolve_exact = create = resume = submit = history = info = interrupt = _refuse


def refused_peer_turn(service, room_id, task):
    """The refusal for a peer turn this v1 cannot deliver, else None."""
    payload = task['payload']
    if payload.get('attachments'):
        return _RefusedTurn('This member is on another gateway and can receive text only.')
    if (room_id, str(payload.get('target_member_id') or payload['target_profile'])) not in service.peer_routes:
        return _RefusedTurn('This member is on another gateway that has not joined this Group Chat yet.')
    return None
