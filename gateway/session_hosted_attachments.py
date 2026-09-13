"""Owner-authorized room byte RPCs and committed task input materialization."""
import base64
import binascii

from gateway.hosted_room_attachments import HostedRoomAttachmentStore, MAX_ATTACHMENT_BYTES
from hermes_state_runtime import RuntimeStoreError


def _authorize(service, actor, params, capability):
    if actor.profile_id != service.authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if capability not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    room_id = params.get('room_id')
    service.authorize_room(actor.subject, room_id)
    service._owned_authority(room_id)
    service._room(room_id)
    return room_id


def upload(service, actor, params):
    room_id = _authorize(service, actor, params, 'session:submit')
    encoded = params.get('data_base64')
    if not isinstance(encoded, str) or len(encoded) > ((MAX_ATTACHMENT_BYTES + 2) // 3) * 4:
        raise RuntimeStoreError('invalid_params')
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RuntimeStoreError('invalid_params') from exc
    return HostedRoomAttachmentStore(service.db_path).put(
        room_id=room_id, upload_id=params.get('upload_id'), kind=params.get('kind'),
        name=params.get('name'), mime=params.get('mime'), data=data)


def download(service, actor, params):
    room_id = _authorize(service, actor, params, 'session:read')
    if not params.get('event_id'):
        raise RuntimeStoreError('invalid_params')
    gateway_id, epoch = service._owned_authority(room_id)
    saved = HostedRoomAttachmentStore(service.db_path).read_viewer(
        room_id=room_id, attachment_id=params.get('attachment_id'), event_id=params['event_id'],
        authority_gateway_id=gateway_id, authority_epoch=epoch)
    return {**saved.attachment, 'data_base64': base64.b64encode(saved.data).decode('ascii')}


def append_user_event(service, *, room_id, event_id, payload, gateway_id, epoch):
    from gateway import hosted_rooms
    store = HostedRoomAttachmentStore(service.db_path)
    manifest = payload.get('attachments', [])
    transitioned = []
    if manifest:
        _, transitioned = store.commit_message_with_receipt(
            room_id=room_id, event_id=event_id, manifest=manifest,
            recipient_member_ids=[m['member_id'] for m in service._room(room_id)['members']],
            viewer_access=True, hold_until_event=True)
    try:
        return hosted_rooms.append_event(
            service.db_path, room_id=room_id, event_id=event_id, kind='message.user',
            actor={'kind': 'user', 'id': 'desktop'}, payload=payload,
            authority_gateway_id=gateway_id, authority_epoch=epoch)
    except Exception:
        if transitioned:
            store.abort_message_commit(room_id=room_id, event_id=event_id, attachment_ids=transitioned)
        raise


def submission_payload(rpc, prompt, attachments=None, *, admission=None):
    """Accepted-input reconstruction only; new inputs require a durable preparation."""
    if admission is None:
        if not attachments:
            return {'text': prompt}
        raise RuntimeStoreError('input_preparation_required')
    return committed_submission_payload(rpc, prompt, attachments, admission=admission)


def committed_submission_payload(rpc, prompt, attachments=None, *, admission):
    from gateway.hosted_room_input_preparation import reconstruct_accepted_payload
    return reconstruct_accepted_payload(rpc, prompt, attachments, admission)
