"""Owner-authorized room byte RPCs and committed task input materialization."""
import base64
import binascii
import os
from pathlib import Path
import tempfile

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


def submission_payload(rpc, prompt, attachments=None):
    """Resolve only event-bound, member-authorized bytes, never a caller path."""
    if not attachments:
        return {'text': prompt}
    from gateway.hosted_room_driver import validate_bound_task_manifest
    from gateway.session_ingress_media import capture_native_media, restore_native_media
    manifest = validate_bound_task_manifest(attachments)
    store = HostedRoomAttachmentStore(rpc.authority.db.db_path)
    references = []
    transferred = getattr(rpc, 'hosted_attachment_data', None)
    if transferred is not None and [item for item, data in transferred] != manifest:
        raise RuntimeStoreError('permission_denied')
    for index, item in enumerate(manifest):
        if transferred is None:
            saved = store.read(room_id=rpc.room_id, attachment_id=item['attachment_id'],
                               event_id=item['event_id'], recipient_member_id=rpc.member_id)
            if any(saved.attachment[key] != item[key] for key in ('kind', 'name', 'mime', 'size')):
                raise RuntimeStoreError('permission_denied')
            data = saved.data
        else:
            data = transferred[index][1]
        if len(data) != item['size']:
            raise RuntimeStoreError('permission_denied')
        # Retained native-inputs are excluded from age-only document cleanup. The
        # content-addressed destination is stable on retry and refuses corruption.
        with tempfile.TemporaryDirectory(prefix='hermes-room-input-') as directory:
            path = Path(directory) / item['name']
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as output:
                output.write(data)
            reference = capture_native_media([path])[0]
        references.append(reference)
    paths = restore_native_media(references)
    from gateway.session_ingress_media import _ATTACHMENT_MIMES
    image_paths = [path for path, item in zip(paths, manifest)
                   if item['mime'] in _ATTACHMENT_MIMES]
    if image_paths:
        # Keep the canonical admission shape; the authority captures these
        # already-authorized cache paths and the normal runner restores them into
        # native image_url parts.
        return {'text': prompt, 'attachments': [
            {'path': path, 'mime': item['mime']}
            for path, item in zip(paths, manifest)
            if item['mime'] in _ATTACHMENT_MIMES
        ]}
    return {'text': prompt + ''.join('\n[Shared attachment] file: ' + path + '\n'
               for path in paths)}
