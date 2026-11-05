"""Read-only first-round projection under the room owner's authority."""
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError


def _member_row(member):
    return {'member_id': member.member_id, 'handle': member.handle,
            **({'display_name': member.display_name} if member.display_name else {})}


def _thread_exists(conn, room_id, thread_id):
    """Whether the room log already holds a message in this thread (the planner's earlier-event scan)."""
    return conn.execute(
        "SELECT 1 FROM hosted_room_events WHERE room_id=? AND kind IN ('message.user', 'message.member') "
        "AND json_extract(payload_json, '$.thread_id')=? LIMIT 1", (room_id, thread_id)).fetchone() is not None


def preview(authority, actor, params):
    from gateway import hosted_room_lead as lead_routing
    from gateway.hosted_room_context import _latest_context, validate_context
    from gateway.hosted_room_discussion import MAX_USER_TEXT_BYTES, resolve_mentions, validate_room
    from gateway.hosted_rooms import _room_from_row
    from gateway.hosted_rooms_common import identifier
    from hermes_cli.install_identity import _read_existing
    from hermes_constants import get_default_hermes_root

    text, room_id = params.get('text'), params.get('room_id')
    thread_id, latest_seq = params.get('thread_id'), params.get('latest_seq')
    if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_USER_TEXT_BYTES:
        raise RuntimeStoreError('invalid_params')
    if thread_id is not None:
        if not isinstance(thread_id, str):
            raise RuntimeStoreError('invalid_params')
        identifier(thread_id, label='thread_id', error=ValueError)
    if isinstance(latest_seq, bool) or not isinstance(latest_seq, int) or latest_seq < 0:
        raise RuntimeStoreError('invalid_params')
    service = getattr(authority, 'hosted_room_service', None)
    authorize = getattr(service, 'authorize_room', None)
    if service is None or not callable(authorize):
        raise RuntimeStoreError('permission_denied')
    if Path(service.db_path).resolve() != Path(authority.db.db_path).resolve():
        raise RuntimeStoreError('profile_mismatch')
    status = service.runtime.status()
    if not status.get('running') or status.get('stopping'):
        raise RuntimeStoreError('runtime_coordination_required')
    install_id, _ = _read_existing(get_default_hermes_root() / 'install_id')
    if not install_id:
        raise RuntimeStoreError('runtime_coordination_required')
    gateway_id = f'install:{install_id}'
    with authority.db._read_ctx(reopen=False) as conn:
        conn.execute('BEGIN')
        try:
            authorize(actor.subject, room_id, conn=conn)
            row = conn.execute('SELECT * FROM hosted_rooms WHERE room_id=?', (room_id,)).fetchone()
            if row is None:
                raise RuntimeStoreError('permission_denied')
            raw = _room_from_row(row)
            if raw['authority_gateway_id'] != gateway_id:
                raise RuntimeStoreError('permission_denied')
            # The composer's observation is the fence: a message or settings event that landed after it
            # makes this projection describe a room the user is not looking at, so fail closed instead of
            # answering from newer state (room revision alone does not move for log events).
            observed_seq = int(row['next_seq']) - 1
            if latest_seq != observed_seq:
                raise RuntimeStoreError('stale_preview')
            # Routing context comes from the same read transaction as the room and the thread scan, so a
            # concurrent settings save cannot pair an old projection with the new routing.
            context = validate_context(**_latest_context(conn, room_id, None))
            room = validate_room(raw, local_profiles=service.room_profiles(raw))
            lead = lead_routing.lead_member(room, context)
            decides = lead is not None and lead_routing.lead_routes_message(
                text=text, room=room,
                # A previewed send without a thread id is the new thread the composer will mint.
                thread_exists=thread_id is not None and _thread_exists(conn, room.room_id, thread_id))
            selected = () if decides else resolve_mentions((text,), room.members)
            return {'room_id': room.room_id, 'authority_gateway_id': room.gateway_id,
                    'authority_epoch': room.authority_epoch, 'revision': raw['revision'],
                    'latest_seq': observed_seq, 'mode': 'lead' if decides else 'members',
                    'members': [] if decides else [_member_row(m) for m in selected],
                    'lead': _member_row(lead) if decides else None}
        finally:
            conn.rollback()
