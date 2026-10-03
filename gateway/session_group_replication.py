"""Canonical controls for Group Chat copies: custody, reading a copy, and retiring it.

Every participant gateway keeps a copy unless its operator mints the member grant with
``replication: false`` (``groups.peer.invite``). The room's home copies the room's history to it
(``gateway/hosted_room_replication.py``); custody (``gateway/hosted_room_custody.py``) records which
installations keep it and which may continue the group. A copy-holding installation shows its
copies read-only through ``groups.list``, ``groups.state`` and ``groups.log`` (``copy: true``), to its
operator or to the room's recorded owner. A copy never runs or recovers work by itself.

Retiring a copy after Disband takes both owners: the room owner prepares the obligation on the
home (``groups.replication.prepare``) and hands its public enrollment to the participant's
operator, who enrolls it (``groups.replication.enroll``) or later withdraws it
(``groups.replication.revoke``). See ``gateway/hosted_room_replica_retirement.py``.
"""
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

TARGET_METHODS = frozenset({
    'groups.replica_state', 'groups.replication.enroll', 'groups.replication.revoke', 'groups.custody.allow'})
CUSTODY_METHODS = frozenset({'groups.custody.designate', 'groups.custody.add', 'groups.custody.remove'})
COPY_READ_METHODS = frozenset({'groups.state', 'groups.log', 'groups.custody.status'})
_OWNER = 'gateway.hosted.owner.v1:'
# A messaging chat other people read (messaging's transport id ``messaging:shared:<grant>``) carries
# the owner's subject, yet it never acts for the owner.
_SHARED_CHAT = 'messaging:shared:'


def refuse_shared_chat(actor):
    """The owner's custody actions never come from a messaging chat other people read."""
    if str(getattr(actor, 'transport_id', '') or '').startswith(_SHARED_CHAT):
        raise RuntimeStoreError('permission_denied')


def dispatch_target(authority, method, params):
    """Participant controls: ``room_id`` names the home's room, which is not a local room here."""
    from gateway import hosted_rooms
    from gateway import hosted_room_custody as custody
    from gateway import hosted_room_replica_retirement as retirement
    from gateway import hosted_room_replicas as replicas
    from gateway.session_authorities import served_profile_name
    # Copies sit beside the default profile's RoomLink grants, as canonical members do.
    if served_profile_name(Path(authority.profile_id)) != 'default':
        raise RuntimeStoreError('default_profile_required')
    db_path = authority.db.db_path
    if method == 'groups.replica_state':
        return replicas.copy_state(db_path, room_id=params.get('room_id'))
    if method == 'groups.custody.allow':
        if set(params) != {'room_id', 'successor'} or type(params['successor']) is not bool:
            raise RuntimeStoreError('invalid_params')
        return custody.set_local_consent(db_path, room_id=hosted_rooms._room_id(params['room_id']),
                                         allowed=params['successor'])
    if method == 'groups.replication.enroll':
        if 'enrollment' not in params:
            raise RuntimeStoreError('invalid_params')
        return retirement.enroll_target(
            db_path, enrollment=params['enrollment'], target_install_id=hosted_rooms.local_authority_gateway_id(),
            expected_enrollment_id=params.get('expected_enrollment_id'),
            expected_state=params.get('expected_state', 'active'))
    if set(params) != {'room_id', 'enrollment_id'}:
        raise RuntimeStoreError('invalid_params')
    return retirement.revoke_target_enrollment(db_path, **params)


def prepare(service, params):
    """The room owner reserves a cleanup obligation for one participant installation."""
    from gateway import hosted_rooms
    from gateway import hosted_room_replica_retirement as retirement
    from gateway.hosted_room_peer import gateway_room_grant_secret
    # Serialized with Disband and route registration, which hold the same lock.
    with service.peer_route_lock:
        enrollment = retirement.prepare_home_enrollment(
            service.db_path, room_id=params.get('room_id'), target_install_id=params.get('target_install_id'),
            endpoint=params.get('endpoint'), local_gateway_id=hosted_rooms.local_authority_gateway_id(),
            secret=gateway_room_grant_secret(), enrollment_id=params.get('enrollment_id'),
            replace_enrollment_id=params.get('replace_enrollment_id'))
    service.replication.wakeup()
    return {'enrollment': enrollment}


# -- the home: custodians and successors -------------------------------------------------------


def _republish(service, room_id):
    """Record the room's custodians in its log at once, let the copies carry it; returns its seq."""
    from contextlib import closing
    from gateway import hosted_room_custody as custody
    from gateway import hosted_rooms
    from gateway.hosted_room_identity import local_public_key
    from gateway.hosted_room_peer import local_room_link_endpoint
    from gateway.hosted_rooms_common import open_sqlite
    endpoint = local_room_link_endpoint()
    name, owner_name = custody.local_names()
    appended = custody.maintain_configuration(
        service.db_path, room_id=room_id, local_gateway_id=hosted_rooms.local_authority_gateway_id(),
        public_key=local_public_key(), endpoint=endpoint.get('url') if endpoint.get('available') else None,
        name=name, owner_name=owner_name)
    service.replication.wakeup()
    if appended is not None:
        return appended['configuration_seq']
    with closing(open_sqlite(service.db_path)) as conn:
        return custody.configuration_locked(conn, room_id)['configuration_seq']


def custody_control(service, method, params):
    """Owner controls on the room's host: designate a successor, add or remove a custodian-only installation."""
    from gateway import hosted_room_custody as custody
    room_id = params.get('room_id')
    service._owned_authority(room_id)
    if method == 'groups.custody.designate':
        if set(params) != {'room_id', 'install_id', 'successor'} or type(params['successor']) is not bool:
            raise RuntimeStoreError('invalid_params')
        row = custody.designate_successor(service.db_path, room_id=room_id, install_id=params['install_id'],
                                          successor=params['successor'])
        return {'room_id': room_id, 'install_id': row['install_id'],
                'successor': bool(row['designated'] and row['allowed'] and row['state'] == 'active'),
                'configuration_seq': _republish(service, room_id)}
    if method == 'groups.custody.remove':
        if set(params) != {'room_id', 'install_id'}:
            raise RuntimeStoreError('invalid_params')
        custody.remove_custody_route(service.db_path, room_id=room_id, install_id=params['install_id'])
        return {'room_id': room_id, 'install_id': params['install_id'],
                'configuration_seq': _republish(service, room_id)}
    return _add_custodian(service, params)


def _add_custodian(service, params):
    """Add an installation that keeps the room's history without a Bot, after a live scoped probe."""
    from gateway import hosted_room_custody as custody
    from gateway.hosted_room_peer import (
        GatewayRoomCatalog, HostedRoomGrantError, HostedRoomPeerError, unverified_room_grant_claims,
        validate_room_link_url)
    from gateway.session_group_peers import probe_route
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    if set(params) - {'successor'} != {'room_id', 'target_url', 'catalog', 'grant'} or type(
            params.get('successor', False)) is not bool or not isinstance(params['grant'], str):
        raise RuntimeStoreError('invalid_params')
    room_id = params['room_id']
    try:
        target_url, _ = validate_room_link_url(params['target_url'])
        catalog = GatewayRoomCatalog.from_mapping(params['catalog'])
        permissions = unverified_room_grant_claims(params['grant']).get('permissions', ())
    except (HostedRoomPeerError, HostedRoomGrantError, TypeError, KeyError) as exc:
        raise RuntimeStoreError('invalid_params') from exc
    gateway_id, epoch = service._owned_authority(room_id)
    install_id = catalog.installation_id
    if install_id == gateway_id or any((m.get('target') or {}).get('installation_id') == install_id
                                       for m in service._room(room_id)['members']):
        raise RuntimeStoreError('peer_target_mismatch')  # its own Bots already make a member installation a custodian
    if 'dispatch' in permissions or 'replicate' not in permissions:
        raise RuntimeStoreError('peer_target_mismatch')
    client = PeerRunsHTTPClient(base_url=target_url, api_key='', receipt_db_path=service.db_path,
                                proof_install_id=install_id)
    probe = probe_route(client, params['grant'], catalog, {
        'room_id': room_id, 'home_install_id': gateway_id, 'authority_gateway_id': gateway_id,
        'authority_epoch': epoch, 'member_id': custody.CUSTODY_MEMBER_ID,
        'target_profile': catalog.execution_policy.target_profile})
    identity = probe.get('room_identity') if isinstance(probe.get('room_identity'), dict) else {}
    identity = identity if identity.get('install_id') == install_id else {}
    with service.peer_route_lock:
        if service._owned_authority(room_id) != (gateway_id, epoch):
            raise RuntimeStoreError('peer_target_mismatch')
        custody.save_custody_route(service.db_path, room_id=room_id, install_id=install_id, target_url=target_url,
                                   target_profile=catalog.execution_policy.target_profile, grant=params['grant'],
                                   catalog=catalog.as_mapping())
        custody.enroll_custodian(
            service.db_path, room_id=room_id, install_id=install_id, public_key=identity.get('public_key'),
            endpoint=target_url, name=identity.get('name'), operator_name=identity.get('operator_name'),
            role='custodian_only', active=True,
            allowed='successor' in permissions or identity.get('allowed') is True,
            designated=params.get('successor', False))
    return {'room_id': room_id, 'install_id': install_id, 'configuration_seq': _republish(service, room_id)}


# -- reading a copy held here ------------------------------------------------------------------


def is_copy(db_path, room_id):
    """Whether this store holds ``room_id`` only as a copy of another gateway's room."""
    from gateway import hosted_room_replicas as replicas
    from gateway import hosted_rooms
    try:
        hosted_rooms.room_state(db_path, room_id=room_id, include_disbanded=True)
        return False
    except hosted_rooms.RoomNotFoundError:
        pass
    except hosted_rooms.HostedRoomError:
        return False
    try:
        state = replicas.replica_state(db_path, room_id=room_id)
    except hosted_rooms.HostedRoomError:
        return False
    return state.get('safety_status') != 'retired'


def may_read_copy(authority, actor, room_id):
    """A copy is shown to this installation's operator, or to the principal recorded as the room's owner."""
    if 'session:operator' in actor.capabilities:
        return True
    with authority.db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_OWNER + room_id,)).fetchone()
    return row is not None and row[0] == actor.subject


def read_copy(authority, actor, method, params):
    """``groups.state``, ``groups.log`` and ``groups.custody.status`` for a copy: read-only, no driver."""
    from gateway import hosted_room_custody as custody
    from gateway import hosted_room_replicas as replicas
    room_id = params['room_id']
    if not may_read_copy(authority, actor, room_id):
        raise RuntimeStoreError('permission_denied')
    db_path = authority.db.db_path
    if method == 'groups.log':
        if set(params) - {'room_id', 'since_seq', 'limit', 'include_disbanded'}:
            raise RuntimeStoreError('invalid_params')
        return replicas.read_copy_events(db_path, room_id=room_id, since_seq=params.get('since_seq', 0),
                                         limit=params.get('limit', 100))
    status = custody.custody_status(db_path, room_id)
    if method == 'groups.custody.status':
        return status
    room = replicas.read_copy_room(db_path, room_id=room_id)
    return {'room': {**room, 'custody': {'configuration_seq': status['configuration_seq'],
                                         'at_risk_after_seq': status['at_risk_after_seq'],
                                         'custodians': status['configuration']['custodians']}},
            'driver_status': None}


def copy_listing(authority, actor):
    """The copies held here that this caller may read, marked ``copy: true``."""
    from gateway import hosted_room_replicas as replicas
    return [copy for copy in replicas.list_copies(authority.db.db_path)
            if may_read_copy(authority, actor, copy['room_id'])]
