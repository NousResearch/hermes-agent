"""Owner-authorized room byte RPCs and committed task input materialization."""
import base64
import binascii
import os
from pathlib import Path
import tempfile

from gateway.hosted_room_attachments import (
    AttachmentAdmissionError, HostedRoomAttachmentStore, MAX_ATTACHMENT_BYTES,
    validate_manifest,
)
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
    # Canonical services reuse their initialized store. Compatibility callers
    # without one may initialize only after principal/room authorization.
    store = getattr(service, 'attachments', None)
    if store is None:
        store = HostedRoomAttachmentStore(service.db_path)
    try:
        return store.put_public(
            room_id=room_id, upload_id=params.get('upload_id'), kind=params.get('kind'),
            name=params.get('name'), mime=params.get('mime'), data=data)
    except AttachmentAdmissionError as exc:
        raise RuntimeStoreError('runtime_coordination_required') from exc


def download(service, actor, params):
    room_id = _authorize(service, actor, params, 'session:read')
    if not params.get('event_id'):
        raise RuntimeStoreError('invalid_params')
    gateway_id, epoch = service._owned_authority(room_id)
    saved = HostedRoomAttachmentStore(service.db_path).read_viewer(
        room_id=room_id, attachment_id=params.get('attachment_id'), event_id=params['event_id'],
        authority_gateway_id=gateway_id, authority_epoch=epoch)
    return {**saved.attachment, 'data_base64': base64.b64encode(saved.data).decode('ascii')}


def append_user_event(
        service, *, room_id, event_id, payload, gateway_id, epoch,
        authorize_new=None, authorize_commit=None, existing_only=False):
    from gateway import hosted_rooms
    manifest = validate_manifest(payload.get('attachments', []))
    store = None
    transitioned = []
    if manifest:
        store = getattr(service, 'attachments', None)
        if store is None:
            store = HostedRoomAttachmentStore(service.db_path)
        _, transitioned = store.commit_message_with_receipt(
            room_id=room_id, event_id=event_id, manifest=manifest,
            recipient_member_ids=[m['member_id'] for m in service._room(room_id)['members']],
            viewer_access=True, hold_until_event=True)
    try:
        return hosted_rooms.append_event(
            service.db_path, room_id=room_id, event_id=event_id, kind='message.user',
            actor={'kind': 'user', 'id': 'desktop'}, payload=payload,
            authority_gateway_id=gateway_id, authority_epoch=epoch,
            authorize_new=authorize_new, authorize_commit=authorize_commit,
            existing_only=existing_only)
    except Exception:
        if transitioned:
            assert store is not None
            store.abort_message_commit(room_id=room_id, event_id=event_id, attachment_ids=transitioned)
        raise


def submission_payload(rpc, prompt, attachments=None):
    """Resolve only event-bound, member-authorized bytes, never a caller path."""
    if not attachments:
        return {'text': prompt}
    from gateway.hosted_room_driver import validate_bound_task_manifest
    from gateway.session_ingress_media import capture_native_media, restore_native_media, validate_media_batch_size
    manifest = validate_bound_task_manifest(attachments)
    # Each bound file is captured on its own, so the admission-wide cap is enforced here,
    # before any member is materialized.
    validate_media_batch_size(item['size'] for item in manifest)
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
    from gateway.platforms.base import get_image_cache_dir
    import hashlib
    images, documents = [], []
    for path, item in zip(paths, manifest):
        if item['mime'] not in _ATTACHMENT_MIMES:
            documents.append(path)
            continue
        # Public admission accepts only owner-local staging paths. Retained
        # room bytes are copied under a deterministic name for exact retries.
        data = Path(path).read_bytes()
        staging = get_image_cache_dir().resolve()
        staging.mkdir(parents=True, exist_ok=True)
        target = staging / (hashlib.sha256(data).hexdigest() + Path(item['name']).suffix)
        if target.exists():
            if target.is_symlink() or target.read_bytes() != data:
                raise RuntimeStoreError('storage_unavailable')
        else:
            fd, temporary = tempfile.mkstemp(dir=staging)
            try:
                with os.fdopen(fd, 'wb') as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        images.append({'path': str(target), 'mime': item['mime']})
    text = prompt + ''.join('\n[Shared attachment] file: ' + path + '\n' for path in documents)
    return {'text': text, **({'attachments': images} if images else {})}


def committed_submission_payload(rpc, prompt, attachments=None):
    from gateway.session_ingress_media import admit_attachments
    payload = submission_payload(rpc, prompt, attachments)
    return {'text': payload['text'], **admit_attachments(payload.get('attachments'))}


def _attested_inputs(attachments, digests):
    """``(manifest item, retained reference)`` per bound input, from source-attested digests.

    Retained inputs are content-addressed under ``native-inputs/<sha256>/<name>`` (images
    under their digest-named staging copy), so the destination path of every input is a
    pure function of manifest and digest; nothing is transferred to compute it.
    """
    from gateway.hosted_room_driver import validate_bound_task_manifest
    from gateway.hosted_room_attachments import _SHA256_RE
    from gateway.session_ingress_media import _ATTACHMENT_MIMES, _media_root
    manifest = validate_bound_task_manifest(attachments)
    if (not isinstance(digests, list) or len(digests) != len(manifest)
            or any(not isinstance(d, str) or _SHA256_RE.fullmatch(d) is None for d in digests)):
        raise RuntimeStoreError('permission_denied')
    root = _media_root()
    inputs = []
    for item, digest in zip(manifest, digests):
        name = digest + Path(item['name']).suffix if item['mime'] in _ATTACHMENT_MIMES else item['name']
        inputs.append((item, {'path': str(root / digest / name), 'sha256': digest, 'size': item['size']}))
    return inputs


def attested_submission_payload(prompt, attachments, digests):
    """The payload ``committed_submission_payload`` commits for these inputs, derived from
    source-attested digests alone.

    The durable row is a pure function of prompt, manifest and per-file digest; a source
    attachment re-pointed at other bytes yields another payload and is refused without
    transferring anything. Images ride as ``attachments_v1`` references that
    ``restore_native_media`` re-verifies at execution; documents ride in the prompt text as
    paths only, so their retained bytes are checked by ``verify_attested_documents``.
    """
    if not attachments:
        return {'text': prompt}
    from gateway.session_ingress_media import _ATTACHMENT_MIMES
    documents, media, media_types = [], [], []
    for item, reference in _attested_inputs(attachments, digests):
        if item['mime'] in _ATTACHMENT_MIMES:
            media.append(reference)
            media_types.append(item['mime'])
        else:
            documents.append(reference['path'])
    text = prompt + ''.join('\n[Shared attachment] file: ' + path + '\n' for path in documents)
    return {'text': text, **({'attachments_v1': {'media': media, 'media_types': media_types}} if media else {})}


def verify_attested_documents(attachments, digests):
    """Refuse ``storage_unavailable`` unless every retained document at the destination
    still hashes to its source-attested digest.

    A document is embedded in the prompt as a content-addressed path, not a structured
    reference, so execution reads whatever bytes sit there; a retained file mutated in
    place or deleted must pause the row here, as ``committed_submission_payload`` did
    before the digest-only preflight. Reads destination bytes only, never the source.
    """
    if not attachments:
        return
    from gateway.session_ingress_media import _ATTACHMENT_MIMES, restore_native_media
    restore_native_media([reference for item, reference in _attested_inputs(attachments, digests)
                          if item['mime'] not in _ATTACHMENT_MIMES])
