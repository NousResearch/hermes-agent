"""Verified RoomLink document bytes enter the existing canonical API custody owner."""
import copy
import json

from hermes_state_runtime import RuntimeStoreError


def prepare(authority, *, session_id, request_id, payload, documents):
    from gateway.hosted_room_input_preparation import PreparedHostedInput, prepare_verified_documents
    settings = payload['api_turn_v1']['settings']
    inputs = settings['room_dispatch']['document_inputs']
    with authority.db._read_ctx() as conn:
        old = conn.execute("SELECT * FROM session_admissions WHERE principal_id='api' AND target_session_id=? AND request_id=?",
                           (session_id, request_id)).fetchone()
    if old is not None:
        saved = json.loads(old['payload_json'])
        comparable = copy.deepcopy(saved)
        comparable['api_turn_v1']['settings'].pop('room_document_inputs', None)
        if comparable != payload:
            raise RuntimeStoreError('admission_conflict')
        return PreparedHostedInput(saved, None)
    if documents is None:
        raise RuntimeStoreError('room_document_input_required')
    if len(documents) != len(inputs) or any(
            (doc['name'], doc['size'], doc['sha256']) != (item['name'], item['size'], item['sha256'])
            for doc, item in zip(documents, inputs)):
        raise RuntimeStoreError('permission_denied')
    def build(references):
        result = copy.deepcopy(payload)
        result['api_turn_v1']['settings']['room_document_inputs'] = {
            'references': list(references), 'request_id': request_id}
        return result
    try:
        return prepare_verified_documents(authority, principal_id='api', session_id=session_id,
            request_id=request_id, documents=documents, build_payload=build)
    except (OSError, ValueError) as exc:
        if isinstance(exc, RuntimeStoreError):
            raise
        raise RuntimeStoreError('storage_unavailable') from exc


def content(authority, ref, payload):
    """Resolve only accepted copy identities, never the transfer or original source."""
    from gateway.hosted_room_documents import manifest
    from gateway.hosted_room_input_reclamation import copy_path, verified_identity
    from gateway.session_admission import admission_fingerprint
    from hermes_state_input_custody import admission_input_refs
    settings = payload['api_turn_v1']['settings']
    dispatch = settings['room_dispatch']
    inputs = manifest(dispatch['document_inputs'], member_id=dispatch['member_id'])
    saved = settings.get('room_document_inputs')
    if not isinstance(saved, dict) or set(saved) != {'references', 'request_id'}:
        raise RuntimeStoreError('storage_unavailable')
    refs = saved['references']
    if not isinstance(refs, list) or len(refs) != len(inputs):
        raise RuntimeStoreError('storage_unavailable')
    digest = admission_fingerprint(canonical_target=ref.session_id, payload={'input': payload, 'intent': 'queue'})
    with authority.db._read_ctx() as conn:
        row = conn.execute("SELECT * FROM session_admissions WHERE principal_id='api' AND target_session_id=? AND request_id=? AND payload_digest=?",
                           (ref.session_id, saved['request_id'], digest)).fetchone()
        if row is None:
            raise RuntimeStoreError('permission_denied')
        copies = admission_input_refs(conn, row) or ()
        if len(copies) != len(inputs):
            raise RuntimeStoreError('storage_unavailable')
        paths = []
        for item, reference, retained in zip(inputs, refs, copies):
            path = copy_path(authority.db, retained)
            if (reference != {'path': str(path), 'sha256': retained['digest'], 'size': retained['size']}
                    or (path.name, retained['digest'], retained['size']) != (item['name'], item['sha256'], item['size'])
                    or verified_identity(path, retained['digest'], retained['size']) != (retained['device'], retained['inode'])):
                raise RuntimeStoreError('storage_unavailable')
            paths.append(str(path))
    return payload['text'] + '\n\nUse the file tools to inspect these shared documents:\n' + '\n'.join(
        f'- {json.dumps(item["name"])}: {json.dumps(path)}' for item, path in zip(inputs, paths))
