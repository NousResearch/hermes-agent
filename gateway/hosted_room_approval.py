"""Authorize one Allow against the room's durable execution and Stop fences."""
from gateway import hosted_room_driver as driver, hosted_rooms as rooms
from hermes_state_runtime import RuntimeStoreError, _epoch


def require_current_approval(service, room_id, member_id, task_id, generation):
    """The short transaction is the source authorization boundary, not delivery.

    A Stop committed before this check refuses Allow. A later Stop may race an
    already-authorized response in transit. Never retain this writer while a
    member RPC or network request runs; local producers recheck at the responder.
    Deny and exact Stop do not need permission to continue execution.
    """
    if type(generation) is not int or generation < 1:
        raise RuntimeStoreError('stale_generation')
    gateway = rooms.local_authority_gateway_id()
    with driver._transaction(service.db_path) as conn:
        authority = getattr(service, 'authority', None)
        if authority is not None:
            _epoch(conn, authority.epoch)
            if conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                            ('gateway.peer.retiring.v1:' + room_id,)).fetchone():
                raise RuntimeStoreError('room_retiring')
        saved = conn.execute('SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?',
                             (room_id, task_id)).fetchone()
        lease = conn.execute('SELECT * FROM hosted_room_driver_leases WHERE room_id=?', (room_id,)).fetchone()
        if saved is None or lease is None:
            raise RuntimeStoreError('stale_generation')
        task = driver._task_from_row(saved)
        payload = task['payload']
        if (task['status'] != 'running' or task['execution_generation'] != generation
                or payload.get('target_member_id', payload['target_profile']) != member_id
                or lease['gateway_id'] != gateway
                or (task['run_gateway_id'], task['run_process_generation'], task['run_lease_generation'])
                != (lease['gateway_id'], lease['process_generation'], lease['lease_generation'])):
            raise RuntimeStoreError('stale_generation')
        driver._require_active_lease(conn, driver._lease_from_row(lease), now=service.runtime.clock())
        room = rooms._room_from_row(conn.execute('SELECT * FROM hosted_rooms WHERE room_id=?',
                                                 (room_id,)).fetchone())
        if not any(member.get('member_id') == member_id and member.get('profile') == payload['target_profile']
                   for member in room['members']):
            raise RuntimeStoreError('permission_denied')
        if conn.execute("SELECT 1 FROM hosted_room_events WHERE room_id=? AND seq>? "
                        "AND kind='room.stop_requested' LIMIT 1", (room_id, payload['source_event_seq'])).fetchone():
            raise RuntimeStoreError('stale_generation')
        return task
