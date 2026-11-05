"""Owner-authorized room byte RPCs and committed task input materialization."""
import base64
import binascii
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

from gateway.hosted_room_attachments import HostedRoomAttachmentStore, MAX_ATTACHMENT_BYTES
from gateway.session_ingress_media import HOSTED_DOCUMENT_LINE
from gateway.session_persistence import authority_persistence
from persistence.core.errors import AuthorityRefusal


def _authorize(service, actor, params, capability):
    if actor.profile_id != service.authority.profile_id:
        raise AuthorityRefusal('profile_mismatch')
    if capability not in actor.capabilities:
        raise AuthorityRefusal('permission_denied')
    room_id = params.get('room_id')
    service.authorize_room(actor.subject, room_id)
    service._owned_authority(room_id)
    service._room(room_id)
    return room_id


@contextmanager
def _room_attachments(service):
    """*service*'s attachment store: the canonical service's on its authority's handle, the process
    service's on a handle opened for the call beside its room file."""
    authority = getattr(service, 'authority', None)
    if authority is not None:
        yield HostedRoomAttachmentStore(authority_persistence(authority))
        return
    from persistence_assembly import open_persistence, profile_scope
    db_path = Path(service.db_path)
    handle = open_persistence(profile_scope(home=db_path.parent))
    try:
        yield HostedRoomAttachmentStore(handle, 'profile' if db_path.name == 'state.db' else 'shared')
    finally:
        handle.close()


def upload(service, actor, params):
    return store_upload(service, _authorize(service, actor, params, 'session:submit'), params)


def download(service, actor, params):
    return read_download(service, _authorize(service, actor, params, 'session:read'), params)


def store_upload(service, room_id, params):
    """Retain one composer file for a later ``groups.send`` manifest (both routes)."""
    encoded = params.get('data_base64')
    if not isinstance(encoded, str) or len(encoded) > ((MAX_ATTACHMENT_BYTES + 2) // 3) * 4:
        raise AuthorityRefusal('invalid_params')
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AuthorityRefusal('invalid_params') from exc
    with _room_attachments(service) as store:
        return store.put(
            room_id=room_id, upload_id=params.get('upload_id'), kind=params.get('kind'),
            name=params.get('name'), mime=params.get('mime'), data=data)


def read_download(service, room_id, params):
    """One committed file for a live room viewer (both routes)."""
    if not params.get('event_id'):
        raise AuthorityRefusal('invalid_params')
    gateway_id, epoch = service._owned_authority(room_id)
    with _room_attachments(service) as store:
        saved = store.read_viewer(
            room_id=room_id, attachment_id=params.get('attachment_id'), event_id=params['event_id'],
            authority_gateway_id=gateway_id, authority_epoch=epoch)
    return {**saved.attachment, 'data_base64': base64.b64encode(saved.data).decode('ascii')}


def append_user_event(service, *, room_id, event_id, payload, gateway_id, epoch):
    from gateway import hosted_rooms
    manifest = payload.get('attachments', [])
    transitioned = []
    with _room_attachments(service) as store:
        store.prepare()
        if manifest:
            _, transitioned = store.commit_message_with_receipt(
                room_id=room_id, event_id=event_id, manifest=manifest,
                recipient_member_ids=[m['member_id'] for m in service._room(room_id)['members']],
                viewer_access=True, hold_until_event=True)
        try:
            return hosted_rooms.append_event(
                service.store, placement=service.placement, room_id=room_id, event_id=event_id, kind='message.user',
                actor={'kind': 'user', 'id': 'desktop'}, payload=payload,
                authority_gateway_id=gateway_id, authority_epoch=epoch)
        except Exception:
            if transitioned:
                store.abort_message_commit(room_id=room_id, event_id=event_id, attachment_ids=transitioned)
            raise


# A file the member can never take: over its media cap, not bound to it, expired or corrupt.
_PERMANENT_FILE_REFUSALS = frozenset({'invalid_params', 'permission_denied'})


def prepare_before_submit(prepare):
    """Run a member turn's file preparation, which precedes its ``prompt.submit``/admission.

    A failure here started no work, so neither route may let the driver book it as uncertain
    execution: a refusal no retry can change settles the turn as failed (``refused_before_submit``)
    and anything else is proven not admitted, so the driver requeues it with backoff."""
    from gateway.hosted_room_attachments import AttachmentError
    from persistence.gateway.hosted_rooms import DriverValidationError
    try:
        return prepare()
    except Exception as exc:
        permanent = isinstance(exc, (AttachmentError, DriverValidationError)) or (
            isinstance(exc, AuthorityRefusal) and exc.reason in _PERMANENT_FILE_REFUSALS)
        if permanent:
            exc.refused_before_submit = True
        else:
            exc.not_admitted = True
        raise


def submission_payload(rpc, prompt, attachments=None):
    """Resolve only event-bound, member-authorized bytes, never a caller path."""
    return bound_submission_payload(
        rpc, room_id=rpc.room_id, member_id=rpc.member_id, prompt=prompt,
        attachments=attachments, transferred=getattr(rpc, 'hosted_attachment_data', None))


def bound_submission_payload(owner, *, room_id, member_id, prompt, attachments=None, transferred=None):
    """``{'text', 'attachments'?}`` for one member turn: documents as prompt path lines, images as
    owner-local staging paths. Shared by both routes, so a file reaches a member identically. *owner*
    reads the files through ``_room_attachments``: the gateway route's RPC through its authority, the
    TUI route through its room file."""
    if not attachments:
        return {'text': prompt}
    from gateway.hosted_room_driver import validate_bound_task_manifest
    from gateway.session_ingress_media import capture_native_media, restore_native_media, validate_media_batch_size
    manifest = validate_bound_task_manifest(attachments)
    # Each bound file is captured on its own, so the admission-wide cap is enforced here,
    # before any member is materialized.
    validate_media_batch_size(item['size'] for item in manifest)
    references = []
    if transferred is not None and [item for item, data in transferred] != manifest:
        raise AuthorityRefusal('permission_denied')
    for index, item in enumerate(manifest):
        if transferred is None:
            with _room_attachments(owner) as store:
                saved = store.read(room_id=room_id, attachment_id=item['attachment_id'],
                                   event_id=item['event_id'], recipient_member_id=member_id)
            if any(saved.attachment[key] != item[key] for key in ('kind', 'name', 'mime', 'size')):
                raise AuthorityRefusal('permission_denied')
            data = saved.data
        else:
            data = transferred[index][1]
        if len(data) != item['size']:
            raise AuthorityRefusal('permission_denied')
        # Retained native-inputs are excluded from age-only document cleanup; a live hosted row
        # holds its documents through the prompt path lines (``hosted_document_references``).
        # The content-addressed destination is stable on retry and refuses corruption.
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
                raise AuthorityRefusal('storage_unavailable')
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
    text = prompt + ''.join(HOSTED_DOCUMENT_LINE + path + '\n' for path in documents)
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
        raise AuthorityRefusal('permission_denied')
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
    text = prompt + ''.join(HOSTED_DOCUMENT_LINE + path + '\n' for path in documents)
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
