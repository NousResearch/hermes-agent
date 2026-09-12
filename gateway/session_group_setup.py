"""Outbound setup receipts on the canonical owner's DB, not execution receipts.

An uncompleted intent is deliberately not retried: a lost remote retirement ACK
cannot be reconstructed. Completion and the exact route share the route writer's
transaction; replay observes that route rather than replaying a network mutation.
"""
import base64
import hashlib
import json
import math
import time
from pathlib import Path

from gateway import hosted_room_links as links, hosted_rooms as rooms
from gateway.hosted_room_peer import GatewayRoomCatalog, PROTOCOL_VERSION, validate_room_link_url
from gateway.hosted_room_route_schema import require_room_work_open
from hermes_state_runtime import RuntimeStoreError, _epoch
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute

_RECEIPT = 'gateway.hosted.peer.setup.v1:'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _text(value, *, limit=256):
    if (not isinstance(value, str) or not value or len(value) > limit
            or value != value.strip() or any(ord(c) < 32 for c in value)):
        raise RuntimeStoreError('invalid_params')
    return value


def _grant_claims(grant):
    # This is only a conservative local scope/lifetime check. Only the target's
    # authenticated scoped probe verifies the signature; we never borrow its key.
    try:
        encoded, signature = grant.split('.')
        if not signature:
            raise ValueError('missing signature')
        claims = json.loads(base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)))
        if not isinstance(claims, dict):
            raise ValueError('invalid claims')
        now = time.time()
        issued, expiry = claims['issued_at'], claims['expires_at']
        if (type(issued) not in {int, float} or type(expiry) not in {int, float}
                or not math.isfinite(issued) or not math.isfinite(expiry)
                or not issued - 30 <= now < expiry
                or not {'dispatch', 'status', 'stop'} <= set(claims['permissions'])):
            raise ValueError('unusable grant')
        return claims
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise RuntimeStoreError('invalid_room_grant') from exc


def _route_record(conn, room_id, member_id):
    row = conn.execute('SELECT * FROM hosted_room_links WHERE room_id=? AND member_id=?',
                       (room_id, member_id)).fetchone()
    return links.StoredRoomLink.from_record(dict(row)).as_record() if row is not None else None


def register_peer(authority, actor, service, params):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    if not isinstance(service, CanonicalHostedRoomService) or service.authority is not authority:
        raise RuntimeStoreError('runtime_coordination_required')
    room_id = _text(params.get('room_id'), limit=128)
    service.authorize_room(actor.subject, room_id)
    request_id = _text(params.get('request_id'))
    member_id, profile = _text(params.get('member_id')), _text(params.get('target_profile'))
    grant = _text(params.get('grant'), limit=links.MAX_GRANT_CHARS)
    url, security = validate_room_link_url(params.get('target_url'))
    catalog = GatewayRoomCatalog.from_mapping(params.get('catalog'))
    if (PROTOCOL_VERSION not in catalog.protocol_versions or 'direct' not in catalog.link_modes
            or not catalog.text or catalog.execution_policy.target_profile != profile
            or catalog.execution_policy.approval_mode == 'off'):
        raise RuntimeStoreError('peer_target_unsupported')
    cancel = _text(params.get('cancellation_scope_id', 'cancel-' + room_id))
    trace = _text(params.get('trace_id', 'setup-' + hashlib.sha256(request_id.encode()).hexdigest()))
    expected = params.get('expected_grant_sha256')
    if 'expected_grant_sha256' in params and (not isinstance(expected, str) or (
            expected != '' and (len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected)))):
        raise RuntimeStoreError('invalid_params')
    home, epoch = authority.profile_id, authority.epoch
    gateway_id = rooms.local_authority_gateway_id()
    key = _RECEIPT + request_id
    intent = dict(operation='groups.peer.register', subject=actor.subject, home=home,
        runtime_epoch=epoch, room_id=room_id, member_id=member_id, target_profile=profile,
        target_url=url, catalog=catalog.as_mapping(), grant_sha256=hashlib.sha256(grant.encode()).hexdigest(),
        cancellation_scope_id=cancel, trace_id=trace, expected_grant_sha256=expected)
    result = dict(registered=True, mode='direct', transport_security=security,
                  target_install_id=catalog.installation_id, target_profile=profile)

    def validate(conn):
        status = service.runtime.status()
        if (authority.hosted_room_service is not service or not status.get('running') or status.get('stopping')):
            raise RuntimeStoreError('runtime_coordination_required')
        if (actor.profile_id != home or authority.profile_id != home
                or Path(service.db_path).resolve() != Path(authority.db.db_path).resolve()
                or Path(service.db_path).resolve().parent != Path(home).resolve()):
            raise RuntimeStoreError('profile_mismatch')
        _epoch(conn, epoch)
        service.authorize_room(actor.subject, room_id, conn=conn)
        require_room_work_open(conn, room_id, error=rooms.HostedRoomError)
        rooms.room_safety._raise_if_quarantined(conn, room_id)
        row = conn.execute('SELECT * FROM hosted_rooms WHERE room_id=?', (room_id,)).fetchone()
        retired = conn.execute('SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?', (room_id,)).fetchone()
        if row is None or row['disbanded_at'] is not None or retired:
            raise RuntimeStoreError('peer_setup_conflict')
        if row['authority_gateway_id'] != gateway_id:
            raise RuntimeStoreError('permission_denied')
        members = json.loads(row['members_json'])
        member = next((m for m in members if m.get('member_id') == member_id), None)
        target = member.get('target', {}) if member else {}
        if (not member or member['profile'] != profile or target.get('kind') != 'peer'
                or target.get('profile') != profile or target.get('installation_id') != catalog.installation_id
                or target.get('capability_digest') != catalog.catalog_digest):
            raise RuntimeStoreError('peer_setup_conflict')
        return dict(authority_epoch=row['authority_epoch'], members=members)

    def receipt(conn):
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def reserve(conn):
        binding = validate(conn)
        fingerprint = _digest({**intent, **binding})
        prior = receipt(conn)
        current = _route_record(conn, room_id, member_id)
        if prior is not None:
            if prior.get('fingerprint') != fingerprint:
                raise RuntimeStoreError('peer_setup_conflict')
            if prior.get('state') != 'completed':
                raise RuntimeStoreError('peer_setup_pending')
            if prior.get('route_digest') != _digest(current):
                raise RuntimeStoreError('peer_setup_conflict')
            _grant_claims(grant)
            return prior, True
        claims = _grant_claims(grant)
        scope = dict(room_id=room_id, home_install_id=gateway_id, authority_gateway_id=gateway_id,
            authority_epoch=binding['authority_epoch'], member_id=member_id, target_profile=profile,
            target_install_id=catalog.installation_id, execution_policy_digest=catalog.execution_policy.policy_digest)
        if (type(claims.get('authority_epoch')) is not int or claims.get('version') != PROTOCOL_VERSION
                or any(claims.get(k) != v for k, v in scope.items())):
            raise RuntimeStoreError('invalid_room_grant')
        current_hash = hashlib.sha256(current['grant'].encode()).hexdigest() if current else ''
        if expected is not None and expected != current_hash:
            raise RuntimeStoreError('peer_setup_conflict')
        pending = dict(fingerprint=fingerprint, state='pending', binding=binding,
                       previous_digest=_digest(current), previous_grant_sha256=current_hash)
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, _json(pending)))
        return pending, False

    pending, replay = authority.db._execute_write(reserve)
    if replay:
        return result

    def guard(conn, record):
        binding = validate(conn)
        if binding != pending['binding'] or receipt(conn) != pending:
            raise RuntimeStoreError('peer_setup_conflict')
        current = _route_record(conn, room_id, member_id)
        if _digest(current) != pending['previous_digest']:
            raise RuntimeStoreError('peer_setup_conflict')
        _grant_claims(grant)
        if record is not None:
            conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                (_json({**pending, 'state': 'completed', 'route_digest': _digest(record)}), key))
        return links.StoredRoomLink.from_record(current) if current is not None else None

    client = PeerRunsHTTPClient(base_url=url, api_key='', target_profile=profile, receipt_db_path=service.db_path)
    try:
        probe = client.probe(grant=grant)
        if GatewayRoomCatalog.from_mapping(probe.get('catalog')) != catalog:
            raise RuntimeStoreError('peer_target_mismatch')
        scope = dict(room_id=room_id, home_install_id=gateway_id, authority_gateway_id=gateway_id,
                     authority_epoch=pending['binding']['authority_epoch'], member_id=member_id, target_profile=profile)
        if type(probe.get('authority_epoch')) is not int or any(probe.get(k) != v for k, v in scope.items()):
            raise RuntimeStoreError('peer_target_mismatch')
        route = PeerMemberRoute(home_install_id=gateway_id, member_id=member_id,
            target_install_id=catalog.installation_id, target_profile=profile,
            capability_digest=catalog.catalog_digest, execution_policy_digest=catalog.execution_policy.policy_digest,
            cancellation_scope_id=cancel, trace_id=trace, grant=grant, attachments=catalog.attachments)
        service.register_peer_route(room_id=room_id, member_id=member_id, route=route, client=client,
            target_url=url, catalog=catalog, expected_grant_sha256=pending['previous_grant_sha256'], setup_guard=guard)
    except RuntimeStoreError:
        raise
    except Exception:
        # Do not expose a raw HTTP body or claim remote retirement was undone.
        raise RuntimeStoreError('peer_setup_pending') from None
    authority.db._execute_write(reserve)  # Read back the exact committed route and receipt.
    return result
