"""Participant YOLO belongs to its canonical owner's existing session route."""
from dataclasses import replace
from functools import partial

from hermes_state_runtime import RuntimeStoreError


def member_yolo(service, *, room_id, member_id, enabled=None):
    from gateway.hosted_rooms import HostedRoomError
    from tui_gateway.hosted_room_driver import HostedRoomBinding

    if not isinstance(member_id, str) or not member_id or (enabled is not None and type(enabled) is not bool):
        raise RuntimeStoreError('invalid_params')
    gateway, epoch = service._owned_authority(room_id)
    room = service._room(room_id)
    member = next((m for m in room['members'] if m['member_id'] == member_id), None)
    if member is None:
        raise RuntimeStoreError('not_found', field='member_id', detail='select a current room member')
    if (member.get('target') or {}).get('kind', 'local') != 'local':
        raise HostedRoomError('participant YOLO is available only for local room members')
    # The same member/profile binding as the producer; never find a session by its display title.
    rpc = service._resolve_member_transport(HostedRoomBinding(room_id, gateway, epoch), {
        'payload': {'target_profile': member['profile'], 'target_member_id': member_id}})
    result = rpc._call('yolo', enabled=enabled)
    return {'room_id': room_id, 'member_id': member_id, **result}


async def session_yolo(rpc, params):
    from gateway.session_policy import policy_for_source
    from gateway.session_runtime_workers import tracked_write

    if set(params) - {'enabled'} or (params.get('enabled') is not None and type(params['enabled']) is not bool):
        raise RuntimeStoreError('invalid_params')
    authority, ref, enabled = rpc.authority, rpc.ref, params.get('enabled')
    # Cold restore loads routing metadata and frozen policy, never an agent or a queued turn.
    authority.authorize(rpc.principal, ref, 'session:control' if enabled is not None else 'session:read')
    live = authority.sessions[ref.session_id]
    policy = policy_for_source(authority.runner, live.source)
    if policy is None or policy.source != 'bot_room':
        raise RuntimeStoreError('permission_denied')
    return await tracked_write(authority, partial(_state, rpc, live.route, enabled, policy.yolo), ordered=True)


def _state(rpc, route, enabled, launch):
    from tools.approval import is_approval_bypass_active_for_session, is_session_yolo_enabled
    from tools.approval_yolo import toggle_session_yolo

    store = rpc.authority.runner.session_store
    with store._lock:
        entry = store._entry_locked(route)
        if entry is None:
            raise RuntimeStoreError('not_found')
        if enabled is not None:
            # Candidate first: a failed write cannot poison either the cached entry or live gate.
            try:
                store._save_entry(route, entry_data=replace(entry, yolo=enabled).to_dict(), lock_held=True)
            except OSError as exc:
                raise RuntimeStoreError('storage_unavailable') from exc
            entry.yolo = enabled
            toggle_session_yolo(route, enabled)
        active = is_session_yolo_enabled(route) or entry.yolo is True or (entry.yolo is None and launch)
        return {'session_key': entry.session_id, 'enabled': active,
                'effective_enabled': active or is_approval_bypass_active_for_session(route)}
