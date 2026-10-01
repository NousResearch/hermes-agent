"""Canonical controls for passive Group Chat copies and their retirement.

A participant gateway opts in when its operator mints a member grant with ``replication: true``
(``groups.peer.invite``). The room's home then copies the room's history to it
(``gateway/hosted_room_replication.py``); the participant's operator reads that copy with
``groups.replica_state``. A copy is installation-wide evidence, stored beside the grants that
admit it, and never confers authority to run or recover work.

Retiring a copy after Disband takes both owners: the room owner prepares the obligation on the
home (``groups.replication.prepare``) and hands its public enrollment to the participant's
operator, who enrolls it (``groups.replication.enroll``) or later withdraws it
(``groups.replication.revoke``). See ``gateway/hosted_room_replica_retirement.py``.
"""
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

TARGET_METHODS = frozenset({'groups.replica_state', 'groups.replication.enroll', 'groups.replication.revoke'})


def dispatch_target(authority, method, params):
    """Participant controls: ``room_id`` names the home's room, which is not a local room here."""
    from gateway import hosted_rooms
    from gateway import hosted_room_replica_retirement as retirement
    from gateway import hosted_room_replicas as replicas
    from gateway.session_authorities import served_profile_name
    # Copies sit beside the default profile's RoomLink grants, as canonical members do.
    if served_profile_name(Path(authority.profile_id)) != 'default':
        raise RuntimeStoreError('default_profile_required')
    db_path = authority.db.db_path
    if method == 'groups.replica_state':
        return replicas.copy_state(db_path, room_id=params.get('room_id'))
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
