"""Read-only first-round projection under the room owner's authority."""
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError


def preview(authority, actor, params):
    from gateway.hosted_room_discussion import MAX_USER_TEXT_BYTES, resolve_mentions, validate_room
    from gateway.hosted_rooms import _room_from_row
    from hermes_cli.install_identity import _read_existing
    from hermes_constants import get_default_hermes_root

    text, room_id = params.get('text'), params.get('room_id')
    if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_USER_TEXT_BYTES:
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
            room = validate_room(raw, local_profiles=service.room_profiles(raw))
            selected = resolve_mentions((text,), room.members)
            return {'room_id': room.room_id, 'authority_gateway_id': room.gateway_id,
                    'authority_epoch': room.authority_epoch, 'revision': raw['revision'],
                    'members': [{'member_id': m.member_id, 'handle': m.handle,
                                 **({'display_name': m.display_name} if m.display_name else {})}
                                for m in selected]}
        finally:
            conn.rollback()
