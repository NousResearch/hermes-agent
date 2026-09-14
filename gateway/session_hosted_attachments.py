"""Owner-authorized room byte RPCs and committed task input materialization."""
import base64
import binascii
import json
from pathlib import Path

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


def attested_submission_payload(prompt, attachments, digests, *, db, admission):
    """Reconstruct the accepted layout from source digests, without another transfer.

    v3 documents use the admission's exact custody references, not native-cache paths.
    This read-only preflight neither prepares input nor repairs missing local bytes.
    Documents embedded in prompt text must be re-hashed here, before execution.
    """
    from gateway.hosted_room_driver import validate_bound_task_manifest
    from gateway.hosted_room_attachments import _SHA256_RE
    from gateway.session_ingress_media import _ATTACHMENT_MIMES, _media_root
    from gateway.hosted_room_input_reclamation import copy_path, verified_identity
    from gateway.session_admission import admission_fingerprint
    from hermes_state_input_custody import admission_input_refs
    manifest = validate_bound_task_manifest(attachments) if attachments else []
    if (not isinstance(digests, list) or len(digests) != len(manifest)
            or any(not isinstance(d, str) or _SHA256_RE.fullmatch(d) is None for d in digests)):
        raise RuntimeStoreError('permission_denied')
    with db._read_ctx() as conn:
        raw = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?',
                           (admission['admission_id'],)).fetchone()
        if raw is None or any(raw[k] != admission[k] for k in
                              ('principal_id', 'target_session_id', 'request_id')):
            raise RuntimeStoreError('permission_denied')
        has_refs = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='input_custody_refs'").fetchone()
        refs = admission_input_refs(conn, raw) if has_refs else None
    document_count = sum(item['mime'] not in _ATTACHMENT_MIMES for item in manifest)
    if refs and len(refs) != document_count:
        raise RuntimeStoreError('storage_unavailable')
    root = _media_root()
    documents, media, media_types = [], [], []
    for item, digest in zip(manifest, digests):
        if item['mime'] in _ATTACHMENT_MIMES:
            path = root / digest / (digest + Path(item['name']).suffix)
            media.append({'path': str(path),
                          'sha256': digest, 'size': item['size']})
            media_types.append(item['mime'])
        else:
            if refs:
                copy = refs[len(documents)]
                if (copy['name'], copy['digest'], copy['size']) != (item['name'], digest, item['size']):
                    raise RuntimeStoreError('admission_conflict')
                path = copy_path(db, copy)
            else:
                path = root / digest / item['name']
            documents.append(str(path))
        verified_identity(path, digest, item['size'])
    text = prompt + ''.join('\n[Shared attachment] file: ' + path + '\n' for path in documents)
    payload = {'text': text, **({'attachments_v1': {'media': media, 'media_types': media_types}} if media else {})}
    stored = json.loads(raw['payload_json'])
    if 'local_operator_v1' in stored:
        payload['local_operator_v1'] = stored['local_operator_v1']
    if admission_fingerprint(canonical_target=raw['target_session_id'],
            payload={'input': payload, 'intent': raw['intent']}) != raw['payload_digest']:
        raise RuntimeStoreError('admission_conflict')
    return payload
