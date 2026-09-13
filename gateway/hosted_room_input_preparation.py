"""Authorized hosted inputs: leased v3 preparation or read-only accepted replay."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.hosted_room_input_custody import _copy, _directory
from gateway.hosted_room_input_reclamation import owned_home, require_initialized, copy_path, verified_identity
from gateway.session_admission import admission_fingerprint
from gateway.session_contract import Submission
from gateway.session_ingress_media import _ATTACHMENT_MIMES, _media_root, _sync_directory, validate_media_batch_size
from hermes_state_input_custody import (AcceptedInputHandle, add_copy, admission_input_refs,
    begin_preparation, finish_preparation, preparation, preparation_copies)
from hermes_state_runtime import RuntimeStoreError, _epoch
from hermes_state_terminal import identity_key, terminal_admission


@dataclass(frozen=True)
class PreparedHostedInput:
    payload: dict
    handle: object | None


def lookup_request(rpc, request_id):
    with rpc.authority.db._read_ctx() as conn:
        row = conn.execute('''SELECT * FROM session_admissions
            WHERE principal_id=? AND target_session_id=? AND request_id=?''',
            (rpc.principal.subject, rpc.ref.session_id, request_id)).fetchone()
        if row is not None:
            return dict(row), False
        key = identity_key(rpc.principal.subject, rpc.ref.session_id, request_id)
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        if saved is None:
            return None, False
        try:
            retired = terminal_admission(conn, json.loads(saved[0]))
            if not isinstance(retired, dict) or tuple(retired.get(k) for k in ('principal_id', 'target_session_id', 'request_id')) != (
                    rpc.principal.subject, rpc.ref.session_id, request_id):
                raise ValueError('wrong retired identity')
        except (ValueError, TypeError):
            raise RuntimeStoreError('storage_unavailable') from None
        return retired, True


def resolve_inputs(rpc, attachments):
    if not attachments:
        return []
    from gateway.hosted_room_driver import validate_bound_task_manifest
    manifest = validate_bound_task_manifest(attachments)
    validate_media_batch_size(item['size'] for item in manifest)
    store = HostedRoomAttachmentStore(rpc.authority.db.db_path)
    transferred = getattr(rpc, 'hosted_attachment_data', None)
    if transferred is not None and [item for item, _ in transferred] != manifest:
        raise RuntimeStoreError('permission_denied')
    inputs = []
    for index, item in enumerate(manifest):
        if transferred is None:
            saved = store.read(room_id=rpc.room_id, attachment_id=item['attachment_id'],
                event_id=item['event_id'], recipient_member_id=rpc.member_id)
            if any(saved.attachment[k] != item[k] for k in ('kind', 'name', 'mime', 'size')):
                raise RuntimeStoreError('permission_denied')
            data = saved.data
        else:
            data = transferred[index][1]
        if len(data) != item['size']:
            raise RuntimeStoreError('permission_denied')
        inputs.append((item, data))
    return inputs


def _image_staging(item, data):
    from gateway.platforms.base import get_image_cache_dir
    path = get_image_cache_dir().resolve() / (hashlib.sha256(data).hexdigest() + Path(item['name']).suffix)
    _directory(path.parent)
    _copy(data, path)
    return {'path': str(path), 'mime': item['mime']}


def reconstruct_accepted_payload(rpc, prompt, attachments, admission, *, retired=False):
    """No staging, preparation or repair. Source bytes prove exact old layout."""
    inputs = resolve_inputs(rpc, attachments)
    db = rpc.authority.db
    with db._read_ctx() as conn:
        raw = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (admission['admission_id'],)).fetchone()
        raw = dict(raw) if raw is not None else terminal_admission(conn, admission['admission_id'])
        if raw is None or any(raw[k] != admission[k] for k in ('principal_id', 'target_session_id', 'request_id')):
            raise RuntimeStoreError('permission_denied')
        has_refs = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='input_custody_refs'").fetchone()
        if retired and has_refs:
            refs = conn.execute('''SELECT c.* FROM input_custody_refs r JOIN input_custody_copies c USING(copy_id)
                WHERE r.admission_id=? ORDER BY r.ordinal''', (raw['admission_id'],)).fetchall()
        else:
            refs = admission_input_refs(conn, raw) if has_refs else None
    docs, images, types = [], [], []
    doc_inputs = [(item, data) for item, data in inputs if item['mime'] not in _ATTACHMENT_MIMES]
    if refs and len(refs) != len(doc_inputs):
        raise RuntimeStoreError('storage_unavailable')
    index = 0
    for item, data in inputs:
        digest = hashlib.sha256(data).hexdigest()
        if item['mime'] in _ATTACHMENT_MIMES:
            path = _media_root() / digest / (digest + Path(item['name']).suffix)
            images.append({'path': str(path), 'sha256': digest, 'size': len(data)})
            types.append(item['mime'])
        else:
            if refs:
                copy = refs[index]
                if (copy['name'], copy['digest'], copy['size']) != (item['name'], digest, len(data)):
                    raise RuntimeStoreError('admission_conflict')
                path = copy_path(db, copy)
            else:
                path = _media_root() / digest / item['name']
            docs.append(str(path))
            index += 1
        if not retired:
            verified_identity(path, digest, len(data))
    payload = {'text': prompt + ''.join('\n[Shared attachment] file: ' + path + '\n' for path in docs)}
    if images:
        payload['attachments_v1'] = {'media': images, 'media_types': types}
    if raw.get('payload_json'):
        stored = json.loads(raw['payload_json'])
        if 'local_operator_v1' in stored:
            payload['local_operator_v1'] = stored['local_operator_v1']
    if admission_fingerprint(canonical_target=raw['target_session_id'], payload={'input': payload, 'intent': raw['intent']}) != raw['payload_digest']:
        raise RuntimeStoreError('admission_conflict')
    return payload


def prepare_hosted_input(rpc, *, request_id, prompt, attachments=None, ttl=300):
    db, epoch = rpc.authority.db, rpc.authority.epoch
    old, retired = lookup_request(rpc, request_id)
    if old is not None:
        payload = reconstruct_accepted_payload(rpc, prompt, attachments, old, retired=retired)
        return PreparedHostedInput({'text': prompt}, AcceptedInputHandle(old['admission_id'], payload))
    inputs = resolve_inputs(rpc, attachments)
    documents = [(item, data) for item, data in inputs if item['mime'] not in _ATTACHMENT_MIMES]
    if not documents:
        images = [_image_staging(item, data) for item, data in inputs]
        return PreparedHostedInput({'text': prompt, **({'attachments': images} if images else {})}, None)
    owned_home(db)

    def plan(conn):
        require_initialized(conn, db)
        handle = begin_preparation(conn, epoch=epoch, principal_id=rpc.principal.subject,
            session_id=rpc.ref.session_id, request_id=request_id, ttl=ttl)
        for index, (item, data) in enumerate(documents):
            add_copy(conn, handle=handle, ordinal=index, name=item['name'],
                digest=hashlib.sha256(data).hexdigest(), size=len(data))
        return handle
    handle = db._execute_write(plan)  # Durable preparation precedes filesystem publication.
    def materialize(conn):
        preparation(conn, epoch=epoch, handle=handle, states=('preparing',))
        copies = preparation_copies(conn, handle)
        paths = []
        for copy, (_, data) in zip(copies, documents):
            if copy['state'] not in {'preparing', 'ready'} or copy['generation'] != copy['item_generation']:
                raise RuntimeStoreError('input_preparation_busy')
            path = copy_path(db, copy)
            _directory(path.parent)
            _copy(data, path)  # Always an independent copy, never a native-cache hardlink.
            identity = verified_identity(path, copy['digest'], copy['size'])
            conn.execute("UPDATE input_custody_copies SET state='ready',device=?,inode=? WHERE copy_id=?",
                         (*identity, copy['copy_id']))
            for directory in (path.parent, path.parent.parent, path.parent.parent.parent):
                _sync_directory(directory)
            paths.append(str(path))
        return paths
    paths = db._execute_write(materialize)
    images = [_image_staging(item, data) for item, data in inputs if item['mime'] in _ATTACHMENT_MIMES]
    payload = {'text': prompt + ''.join('\n[Shared attachment] file: ' + path + '\n' for path in paths),
               **({'attachments': images} if images else {})}
    from gateway.session_submission_payload import normalize_submission_payload
    normalized = normalize_submission_payload(rpc.authority, rpc.principal, Submission(request_id, rpc.ref, payload, 'queue'))
    digest = admission_fingerprint(canonical_target=rpc.ref.session_id, payload={'input': normalized, 'intent': 'queue'})
    db._execute_write(lambda conn: finish_preparation(conn, epoch=epoch, handle=handle, payload_digest=digest))
    return PreparedHostedInput(payload, handle)
