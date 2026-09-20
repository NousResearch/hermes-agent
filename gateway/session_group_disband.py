"""Canonical cleanup-before-revoke, replayable under the original owner only.

Restores the O4fb3f28 retire_and_disband ordering (David Dudok de Wit),
using accepted Stop/Output consumers rather than reopening foreign outboxes.
"""
from contextlib import nullcontext
import json
import time
from typing import Any

from gateway import hosted_rooms as rooms
from gateway.hosted_room_links import route_security_digest
from gateway.session_group_state import GroupStateOwner
from gateway.session_group_retirement import require_room_retired
from hermes_state_errors import StateDbCorruptError, StateDbReplacedError
from hermes_state_runtime import RuntimeStoreError
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

PREFIX = 'gateway.hosted.disband.v1:'


def _load(conn, room_id) -> dict[str, Any] | None:
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (PREFIX + room_id,)).fetchone()
    return json.loads(row[0]) if row else None


def sealed_output(service, conn, room_id):
    saved = _load(conn, room_id)
    if saved is None:
        return False
    service.require_output_retirement(conn, room_id, saved['output'])
    return True


def disband(authority, actor, service, params, owner):
    room_id = rooms._room_id(params.get('room_id'))
    if owner is None or owner.authority is not authority:
        raise RuntimeStoreError('group_state_unavailable')
    # The native connection pins the original owner. A request may not silently
    # bind a replacement service/runtime installed while its peer I/O is held.
    installed = getattr(authority, 'hosted_room_service', None)
    if installed is not owner.service:
        raise RuntimeStoreError('group_state_unavailable')
    runtime = owner.runtime
    cancel_id = params.get('cancel_id') or 'room-disbanded'
    metadata_revision = None

    def current(conn, expected=None, *, require_runtime=True):
        owner.current(conn)
        if (actor.profile_id != owner.profile_id or 'session:control' not in actor.capabilities
                or authority.runner.session_authority is not authority
                or getattr(authority, 'hosted_room_service', None) is not installed):
            raise RuntimeStoreError('permission_denied')
        authority._require_admission_open()
        if installed is not None:
            if installed.authority is not authority or installed.runtime is not runtime:
                raise RuntimeStoreError('group_state_unavailable')
            if require_runtime:
                status = runtime.status()
                if not status['running'] or status['stopping']:
                    raise RuntimeStoreError('runtime_coordination_required')
        owned = conn.execute('SELECT value FROM state_meta WHERE key=?', ('gateway.hosted.owner.v1:' + room_id,)).fetchone()
        if owned is None or owned[0] != actor.subject:
            raise RuntimeStoreError('permission_denied')
        rooms.room_safety._raise_if_quarantined(conn, room_id)
        row = conn.execute('SELECT authority_gateway_id,authority_epoch,members_json,disbanded_at FROM hosted_rooms WHERE room_id=?', (room_id,)).fetchone()
        if row is None:
            raise RuntimeStoreError('group_state_unavailable')
        identity = dict(owner=owned[0], epoch=owner.epoch, instance=owner.instance_id,
                        profile=owner.profile_id, gateway=row['authority_gateway_id'],
                        room_epoch=row['authority_epoch'], roster=row['members_json'])
        if (identity['gateway'] != rooms.local_authority_gateway_id()
                or (expected is not None and identity != expected)):
            raise RuntimeStoreError('group_state_unavailable')
        return identity, row['disbanded_at']

    def write(operation):
        # Existing owner writer, not pathname recovery. No I/O is inside it.
        try:
            with owner.db.live_write_connection() as conn:
                return operation(conn)
        except (StateDbCorruptError, StateDbReplacedError) as exc:
            raise RuntimeStoreError('group_state_unavailable') from exc

    lock = installed._output_room_lock(room_id) if installed is not None else nullcontext()
    with lock:
        with owner.read() as conn:
            identity, tombstoned = current(conn, require_runtime=service is not None)
            saved = _load(conn, room_id)
        if tombstoned is not None:
            def replay(conn):
                current(conn, identity, require_runtime=False)
                return rooms.disband_room(owner.db.db_path, room_id=room_id,
                    expected_gateway_id=identity['gateway'], expected_epoch=identity['room_epoch'], conn=conn)
            return {'tombstone': write(replay)}
        if saved is None:
            # Begin is already idempotent and commits before Stop/peer I/O.
            if service is not None:
                service.begin_room_disband(room_id)
                service.stop_room(room_id, cancel_id=cancel_id, require_acknowledged=True)
            else:
                from gateway.hosted_room_link_records import begin_room_link_retirement
                begin_room_link_retirement(owner.db.db_path, room_id=room_id,
                    authority_gateway_id=identity['gateway'], authority_epoch=identity['room_epoch'])
            if service is None:
                def seal_metadata(conn):
                    current(conn, identity, require_runtime=False)
                    require_room_retired(conn, room_id, unavailable=True)
                    if conn.execute(
                            'SELECT 1 FROM hosted_room_links WHERE room_id=?',
                            (room_id,)).fetchone() is not None:
                        raise RuntimeStoreError('runtime_coordination_required')
                    row = conn.execute(
                        'SELECT next_seq FROM hosted_rooms WHERE room_id=?', (room_id,)).fetchone()
                    if row is None:
                        raise RuntimeStoreError('group_state_unavailable')
                    return int(row[0])
                metadata_revision = write(seal_metadata)
            else:
                def seal_output(conn):
                    current(conn, identity)
                    links = [dict(r) for r in conn.execute(
                        'SELECT * FROM hosted_room_links WHERE room_id=? ORDER BY member_id',
                        (room_id,))]
                    output = service.capture_output_retirement(conn, room_id)
                    record: dict[str, Any] = dict(version=1, identity=identity, output=output,
                        revision=conn.execute(
                            'SELECT next_seq FROM hosted_rooms WHERE room_id=?',
                            (room_id,)).fetchone()[0],
                        routes={r['member_id']: route_security_digest(r) for r in links}, revoked=[])
                    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                                 (PREFIX + room_id, json.dumps(record, sort_keys=True)))
                    return record
                saved = write(seal_output)
        elif saved['identity'] != identity:
            raise RuntimeStoreError('group_state_unavailable')

        def validate(conn):
            current(conn, identity, require_runtime=service is not None)
            if saved is None:
                require_room_retired(conn, room_id, unavailable=True)
                row = conn.execute(
                    'SELECT next_seq FROM hosted_rooms WHERE room_id=?', (room_id,)).fetchone()
                if (metadata_revision is None or row is None
                        or int(row[0]) != metadata_revision):
                    raise RuntimeStoreError('group_state_unavailable')
            else:
                if service is None or _load(conn, room_id) != saved:
                    raise RuntimeStoreError('runtime_coordination_required')
                if conn.execute('SELECT next_seq FROM hosted_rooms WHERE room_id=?', (room_id,)).fetchone()[0] != saved['revision']:
                    raise RuntimeStoreError('group_state_unavailable')
                service.require_output_retirement(conn, room_id, saved['output'])
            links = {r['member_id']: dict(r) for r in conn.execute('SELECT * FROM hosted_room_links WHERE room_id=?', (room_id,))}
            if {k: route_security_digest(v) for k, v in links.items()} != (saved['routes'] if saved else {}):
                raise RuntimeStoreError('peer_setup_conflict')
            return links

        with owner.read() as conn:
            links = validate(conn)
        for member, link in links.items():
            if saved is None:
                raise RuntimeStoreError('runtime_coordination_required')
            if member in saved['revoked']:
                continue
            with owner.read() as conn:
                validate(conn)
            client = PeerRunsHTTPClient(base_url=link['target_url'], api_key='', target_profile=link['target_profile'])
            try:
                reply = client.revoke_grant_exact(grant=link['grant'])
            except PeerRunsHTTPError as exc:
                raise RuntimeStoreError('runtime_coordination_required') from exc
            if reply != {'object': 'hermes.room_member.grant.revocation', 'revoked': True} or type(reply.get('revoked')) is not bool:
                raise RuntimeStoreError('runtime_coordination_required')
            current_saved = saved
            def complete_member(conn):
                validate(conn)
                updated = dict(current_saved, revoked=[*current_saved['revoked'], member])
                conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                    (json.dumps(updated, sort_keys=True), PREFIX + room_id))
                return updated
            saved = write(complete_member)

        def finish(conn):
            validate(conn)
            if saved and set(saved['revoked']) != set(saved['routes']):
                raise RuntimeStoreError('runtime_coordination_required')
            cursor = conn.execute('UPDATE hosted_room_disband_fences SET revocation_complete_at=COALESCE(revocation_complete_at,?) '
                'WHERE room_id=? AND authority_gateway_id=? AND authority_epoch=?',
                (time.time(), room_id, identity['gateway'], identity['room_epoch']))
            if cursor.rowcount != 1:
                raise RuntimeStoreError('group_state_unavailable')
            conn.execute('DELETE FROM hosted_room_links WHERE room_id=?', (room_id,))
            result = rooms.disband_room(owner.db.db_path, room_id=room_id,
                expected_gateway_id=identity['gateway'], expected_epoch=identity['room_epoch'], conn=conn)
            # Home publication gets its existing retention grace, not source discard.
            from gateway.hosted_room_attachments import DISBANDED_GRACE_SECONDS
            now = time.time()
            if conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_attachments'"
                    ).fetchone() is not None:
                conn.execute("UPDATE hosted_room_attachments SET state='disbanded',updated_at=?,expires_at=? "
                             "WHERE room_id=? AND state!='disbanded'", (now, now + DISBANDED_GRACE_SECONDS, room_id))
            return result
        result = write(finish)
        if service is not None:
            with service._policy_lock:
                for member in links:
                    key = room_id, member
                    for mapping in (service.peer_routes, service.peer_clients, service._peer_route_status, service._peer_renewals):
                        mapping.pop(key, None)
                    service._persisted_peer_route_keys.discard(key)
                service._peer_renewal_scans.pop(room_id, None)
        return {'tombstone': result}
