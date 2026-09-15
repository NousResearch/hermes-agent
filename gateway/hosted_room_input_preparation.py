"""Authorized hosted inputs: leased v3 preparation or read-only accepted replay."""
from contextlib import contextmanager
from dataclasses import dataclass
import time
import uuid
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
    inputs = [(item, hashlib.sha256(data).hexdigest()) for item, data in resolve_inputs(rpc, attachments)]
    return _reconstruct_payload(rpc.authority.db, prompt, inputs, admission, retired=retired)


def reconstruct_attested_payload(db, prompt, attachments, source_digests, admission):
    """Q's source-attested digest path: no RPC principal and no recapture."""
    from gateway.hosted_room_driver import validate_bound_task_manifest
    from gateway.hosted_room_attachments import _SHA256_RE
    manifest = validate_bound_task_manifest(attachments) if attachments else []
    if (not isinstance(source_digests, list) or len(source_digests) != len(manifest)
            or any(not isinstance(d, str) or _SHA256_RE.fullmatch(d) is None for d in source_digests)):
        raise RuntimeStoreError('permission_denied')
    validate_media_batch_size(item['size'] for item in manifest)
    return _reconstruct_payload(db, prompt, list(zip(manifest, source_digests)), admission)


def _reconstruct_payload(db, prompt, inputs, admission, *, retired=False):
    with db._read_ctx() as conn:
        raw = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (admission['admission_id'],)).fetchone()
        raw = dict(raw) if raw is not None else terminal_admission(conn, admission['admission_id'])
        if raw is None or any(raw[k] != admission[k] for k in ('principal_id', 'target_session_id', 'request_id')):
            raise RuntimeStoreError('permission_denied')
        has_refs = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='input_custody_refs'").fetchone()
        # Lifecycle comes only from durable authority, never the caller's hint.
        terminal = raw['status'] == 'terminal'
        refs = admission_input_refs(conn, raw, require_ready=not terminal) if has_refs else None
    docs, images, types = [], [], []
    doc_inputs = [(item, digest) for item, digest in inputs if item['mime'] not in _ATTACHMENT_MIMES]
    if refs and len(refs) != len(doc_inputs):
        raise RuntimeStoreError('storage_unavailable')
    index = 0
    for item, digest in inputs:
        if item['mime'] in _ATTACHMENT_MIMES:
            path = _media_root() / digest / (digest + Path(item['name']).suffix)
            images.append({'path': str(path), 'sha256': digest, 'size': item['size']})
            types.append(item['mime'])
        else:
            if refs:
                copy = refs[index]
                if (copy['name'], copy['digest'], copy['size']) != (item['name'], digest, item['size']):
                    raise RuntimeStoreError('admission_conflict')
                path = copy_path(db, copy)
            else:
                path = _media_root() / digest / item['name']
            docs.append(str(path))
            index += 1
        if not terminal:
            verified_identity(path, digest, item['size'])
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


def prepare_verified_documents(authority, *, principal_id, session_id, request_id,
                               documents, build_payload, ttl=300):
    """Private trusted seam for verified DOCUMENT bytes, not a peer/API adapter.

    documents: ordered {name, data: bytes, sha256, size} dictionaries. The real
    receiving authority owns the current Home/session. Its caller has already
    authorized source/event/recipient and validated the complete TASK manifest
    (including images and aggregate bytes), and must check accepted replay first.
    build_payload(tuple_of_references) runs once, after durable publication, and
    returns the COMPLETE final admission payload (including api_turn_v1 for API).
    Never serialize the returned handle. Admission still owns grant authorization.
    Images continue through the existing canonical native-media owner, not v3.
    """
    from gateway.hosted_room_attachments import (
        _name, MAX_TASK_ATTACHMENTS, MAX_ATTACHMENT_BYTES, MAX_TASK_ATTACHMENT_BYTES,
    )
    from hermes_state_runtime import _json, _session, _text
    db, epoch = authority.db, authority.epoch
    authority._require_admission_open()
    owned_home(db)
    for value in (principal_id, session_id, request_id):
        _text(value)
    if not isinstance(documents, (list, tuple)) or not 0 < len(documents) <= MAX_TASK_ATTACHMENTS:
        raise RuntimeStoreError('invalid_params')
    if not callable(build_payload):
        raise RuntimeStoreError('invalid_params')
    documents = tuple(dict(item) for item in documents)
    for item in documents:
        if (set(item) != {'name', 'data', 'sha256', 'size'}
                or type(item['data']) is not bytes or type(item['size']) is not int
                or not 0 < item['size'] <= MAX_ATTACHMENT_BYTES
                or _name(item['name']) != item['name']
                or len(item['data']) != item['size']
                or hashlib.sha256(item['data']).hexdigest() != item['sha256']):
            raise RuntimeStoreError('permission_denied')
    if sum(item['size'] for item in documents) > MAX_TASK_ATTACHMENT_BYTES:
        raise RuntimeStoreError('invalid_params')
    validate_media_batch_size(item['size'] for item in documents)

    def plan(conn):
        _epoch(conn, epoch)
        _session(conn, session_id)
        require_initialized(conn, db)
        if (conn.execute('''SELECT 1 FROM session_admissions WHERE principal_id=?
                AND target_session_id=? AND request_id=?''', (principal_id, session_id, request_id)).fetchone()
                or conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                    (identity_key(principal_id, session_id, request_id),)).fetchone()):
            raise RuntimeStoreError('input_preparation_required')
        handle = begin_preparation(conn, epoch=epoch, principal_id=principal_id,
            session_id=session_id, request_id=request_id, ttl=ttl)
        for index, item in enumerate(documents):
            add_copy(conn, handle=handle, ordinal=index, name=item['name'],
                digest=item['sha256'], size=item['size'])
        return handle
    handle = db._execute_write(plan)
    def materialize(conn):
        preparation(conn, epoch=epoch, handle=handle, states=('preparing',))
        copies = preparation_copies(conn, handle)
        paths = []
        for copy, item in zip(copies, documents):
            data = item['data']
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
            paths.append({'path': str(path), 'sha256': copy['digest'], 'size': copy['size']})
        return paths
    paths = db._execute_write(materialize)
    with native_preparation_capture(authority, handle):
        payload = json.loads(_json(build_payload(tuple(paths))))
    digest = admission_fingerprint(canonical_target=session_id, payload={'input': payload, 'intent': 'queue'})
    db._execute_write(lambda conn: finish_preparation(conn, epoch=epoch, handle=handle, payload_digest=digest))
    return PreparedHostedInput(payload, handle)


@contextmanager
def native_preparation_capture(authority, handle):
    """Register canonical native aliases before publication, under the document lease.

    The capture callback is synchronous and scoped to this context/thread. Neither
    a rollback nor a callback exception can roll back the committed capture intent.
    Collection of native aliases remains exclusive/pre-ingress, never online.
    """
    from gateway.session_ingress_media import _preparation_capture
    from hermes_state_input_custody import PreparedInputHandle, copy_is_held
    if handle is None:
        yield
        return
    if not isinstance(handle, PreparedInputHandle):
        raise RuntimeStoreError('invalid_params')
    db, epoch = authority.db, authority.epoch
    owned_home(db)

    def capture(staged, references, publish):
        def plan(conn):
            require_initialized(conn, db)
            preparation(conn, epoch=epoch, handle=handle)
            for temporary, reference in zip(staged, references):
                path = Path(reference['path'])
                identity = verified_identity(path if path.exists() else temporary,
                    reference['sha256'], reference['size'])
                row = conn.execute("SELECT * FROM input_custody_copies WHERE namespace='native' AND digest=? AND name=?",
                    (reference['sha256'], path.name)).fetchone()
                if row is None:
                    copy_id, generation = uuid.uuid4().hex, 1
                    conn.execute('INSERT INTO input_custody_copies VALUES(?,?,?,?,?,?,?,?,?)',
                        (copy_id, 'native', path.name, reference['sha256'], reference['size'], generation, 'ready', *identity))
                else:
                    copy_id, generation = row['copy_id'], row['generation']
                    if row['state'] == 'sealed' or row['size'] != reference['size']:
                        raise RuntimeStoreError('input_preparation_busy')
                    if row['state'] == 'removed' or (row['device'], row['inode']) != identity:
                        if copy_is_held(conn, row, time.time()):
                            raise RuntimeStoreError('input_preparation_busy')
                        generation += 1
                        conn.execute("UPDATE input_custody_copies SET generation=?,state='ready',device=?,inode=? WHERE copy_id=?",
                            (generation, *identity, copy_id))
                conn.execute('INSERT OR IGNORE INTO input_custody_native_items VALUES(?,?,?)',
                    (handle.preparation_id, copy_id, generation))
        # The intent must commit before publication; a failed second transaction
        # leaves restart-safe leased collector work, not unowned durable bytes.
        db._execute_write(plan)
        def materialize(conn):
            preparation(conn, epoch=epoch, handle=handle)
            publish()
        db._execute_write(materialize)
    token = _preparation_capture.set(capture)
    try:
        yield
    finally:
        _preparation_capture.reset(token)


def prepare_hosted_input(rpc, *, request_id, prompt, attachments=None, ttl=300):
    old, retired = lookup_request(rpc, request_id)
    if old is not None:
        payload = reconstruct_accepted_payload(rpc, prompt, attachments, old, retired=retired)
        return PreparedHostedInput({'text': prompt}, AcceptedInputHandle(old['admission_id'], payload))
    inputs = resolve_inputs(rpc, attachments)
    documents = [(item, data) for item, data in inputs if item['mime'] not in _ATTACHMENT_MIMES]
    if not documents:
        images = [_image_staging(item, data) for item, data in inputs]
        return PreparedHostedInput({'text': prompt, **({'attachments': images} if images else {})}, None)
    public_payload = {}
    def final_payload(references):
        images = [_image_staging(item, data) for item, data in inputs if item['mime'] in _ATTACHMENT_MIMES]
        public_payload.update({'text': prompt + ''.join(
            '\n[Shared attachment] file: ' + ref['path'] + '\n' for ref in references),
            **({'attachments': images} if images else {})})
        from gateway.session_submission_payload import normalize_submission_payload
        return normalize_submission_payload(rpc.authority, rpc.principal,
            Submission(request_id, rpc.ref, public_payload, 'queue'))
    prepared = prepare_verified_documents(rpc.authority, principal_id=rpc.principal.subject,
        session_id=rpc.ref.session_id, request_id=request_id, ttl=ttl,
        documents=[{'name': item['name'], 'data': data, 'sha256': hashlib.sha256(data).hexdigest(),
                    'size': len(data)} for item, data in documents], build_payload=final_payload)
    return PreparedHostedInput(public_payload, prepared.handle)
