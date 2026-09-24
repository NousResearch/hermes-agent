"""Files-owned bridge from verified RoomLink staging to canonical API custody.

Documents are independent G v3 copies. Images retain the existing native-media
owner. Neither the spool nor caller JSON is a durable execution reference.
"""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile

from gateway.hosted_room_peer import canonical_attachment_manifest, attachment_manifest_digest
from gateway.session_ingress_media import capture_native_media, restore_native_media, validate_media_batch_size
from hermes_state_runtime import RuntimeStoreError


def peer_input_available(adapter, profile='default', *, connection=None):
    from gateway.session_peer_route import room_route_ready
    return room_route_ready(adapter, connection=connection) and peer_input_initialized(adapter, profile, connection=connection)


def peer_input_initialized(adapter, profile='default', *, connection=None):
    """Read-only readiness of the actual owned, initialized receiver contracts."""
    from gateway.session_peer_target import root_target
    from gateway.hosted_room_input_reclamation import owned_home, require_initialized
    try:
        authority, _ = root_target(adapter, profile, connection=connection)
        owned_home(authority.db)
        if connection is None:
            with authority.db.live_read_connection() as conn:
                if conn is None:
                    return False
                require_initialized(conn, authority.db)
        else:
            require_initialized(connection, authority.db)
        routes = {(method, path) for method, path, handler in adapter._http_route_table() if callable(handler)}
        return {('POST', '/v1/room-members/attachments'),
                ('PUT', '/v1/room-members/attachments/{task_id}/{execution_generation}/{attachment_id}'),
                ('DELETE', '/v1/room-members/attachments/{task_id}/{execution_generation}'),
                ('POST', '/v1/runs')} <= routes
    except (RuntimeStoreError, sqlite3.Error, AttributeError, OSError):
        return False


def _without_media(payload):
    result = copy.deepcopy(payload)
    turn = result.get('api_turn_v1')
    if not isinstance(turn, dict) or not isinstance(turn.get('settings'), dict):
        # Terminal metadata retains identity/digest, not the erased private
        # payload. Only the durable HTTP receipt can resolve that replay.
        raise RuntimeStoreError('storage_unavailable')
    turn['settings'].pop('room_input_media', None)
    return result


def prepare_peer_input(authority, *, session_id, request_id, dispatch, payload):
    """Normalize Files preparation failures for accepted-row-aware Run cleanup."""
    try:
        return _prepare_peer_input(authority, session_id=session_id, request_id=request_id,
                                   dispatch=dispatch, payload=payload)
    except RuntimeStoreError:
        # RuntimeStoreError is a ValueError too; preserve replay/auth reasons.
        raise
    except (OSError, ValueError) as exc:
        # Includes spool initialization/reverification and image/document I/O.
        # The Run owner decides whether admission committed; never delete here.
        raise RuntimeStoreError('storage_unavailable') from exc


def _prepare_peer_input(authority, *, session_id, request_id, dispatch, payload):
    """Bind the COMPLETE final API payload before admission; handle stays private."""
    from gateway.hosted_room_input_preparation import PreparedHostedInput, prepare_verified_documents
    from gateway.platforms.api_server_room_attachments import _request_spool
    from hermes_state_terminal import identity_key, terminal_admission
    # Replay precedes every spool read or new preparation, including direct API retries.
    with authority.db._read_ctx() as conn:
        old = conn.execute('''SELECT * FROM session_admissions WHERE principal_id='api'
            AND target_session_id=? AND request_id=?''', (session_id, request_id)).fetchone()
        if old is None:
            saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                (identity_key('api', session_id, request_id),)).fetchone()
            old = terminal_admission(conn, json.loads(saved[0])) if saved else None
        if old is not None:
            saved_payload = json.loads(old['payload_json'])
            if _without_media(saved_payload) != _without_media(payload):
                raise RuntimeStoreError('admission_conflict')
            return PreparedHostedInput(saved_payload, None)
    spool = _request_spool()
    items = spool.materialize(dispatch)
    manifest = canonical_attachment_manifest([
        {key: value for key, value in item.items() if key != 'path'} for item in items])
    if attachment_manifest_digest(manifest) != dispatch.attachment_manifest_digest:
        raise RuntimeStoreError('permission_denied')
    validate_media_batch_size(item['size'] for item in manifest)
    # Verify the whole batch before publishing the first private document/image.
    verified = [spool._read_verified(Path(item['path']), size=item['size'], digest=item['sha256'])
                for item in items]
    documents = [{'name': item['name'], 'size': item['size'], 'sha256': item['sha256'], 'data': data}
                 for item, data in zip(items, verified) if item['kind'] != 'image']
    images = {}
    for index, (item, data) in enumerate(zip(items, verified)):
        if item['kind'] == 'image':
            with tempfile.TemporaryDirectory(prefix='hermes-peer-input-') as directory:
                path = Path(directory) / item['name']
                path.write_bytes(data)
                images[index] = capture_native_media([path])[0]
    def build(references):
        result = copy.deepcopy(payload)
        docs = iter(references)
        media = [images[index] if item['kind'] == 'image' else next(docs)
                 for index, item in enumerate(items)]
        result['api_turn_v1']['settings']['room_input_media'] = {'manifest': manifest, 'media': media, 'request_id': request_id}
        return result
    if documents:
        return prepare_verified_documents(authority, principal_id='api', session_id=session_id,
            request_id=request_id, documents=documents, build_payload=build)
    return PreparedHostedInput(build(()), None)


def check_peer_input(settings):
    data = settings.get('room_input_media')
    dispatch = settings.get('room_dispatch') or {}
    if data is None:
        if dispatch.get('attachment_manifest_digest') is not None:
            raise RuntimeStoreError('storage_unavailable')
        return [], []
    if (not isinstance(data, dict) or set(data) != {'manifest', 'media', 'request_id'}
            or not isinstance(data.get('request_id'), str) or not data['request_id']):
        raise RuntimeStoreError('invalid_params')
    manifest = canonical_attachment_manifest(data['manifest'])
    references = data['media']
    if (not isinstance(references, list) or len(references) != len(manifest)
            or attachment_manifest_digest(manifest) != dispatch.get('attachment_manifest_digest')):
        raise RuntimeStoreError('permission_denied')
    for reference, item in zip(references, manifest):
        if (not isinstance(reference, dict) or set(reference) != {'path', 'size', 'sha256'}
                or not isinstance(reference['path'], str)
                or Path(reference['path']).name != item['name']
                or (reference['size'], reference['sha256']) != (item['size'], item['sha256'])):
            raise RuntimeStoreError('permission_denied')
    return manifest, references


def peer_input_transcript(payload):
    """Project only frozen prompt + bounded labels, never retained paths/bytes.

    Called after peer_input_content verifies the accepted payload and complete
    retained batch. JSON quoting keeps control characters in basenames inert;
    the manifest bounds the count and name length (no mutable catalog reads).
    """
    manifest, _ = check_peer_input(payload['api_turn_v1']['settings'])
    labels = [f"[Attached {item['kind']}: {json.dumps(item['name'])}]" for item in manifest]
    return payload['text'] + ('\n\n' + '\n'.join(labels) if labels else '')


def peer_input_content(authority, ref, payload):
    from gateway.session_admission import admission_fingerprint
    from gateway.hosted_room_input_reclamation import copy_path, verified_identity
    from hermes_state_input_custody import admission_input_refs
    settings = payload['api_turn_v1']['settings']
    manifest, references = check_peer_input(settings)
    text = payload['text']
    if not manifest:
        return text
    digest = admission_fingerprint(canonical_target=ref.session_id, payload={'input': payload, 'intent': 'queue'})
    with authority.db._read_ctx() as conn:
        admission = conn.execute('''SELECT * FROM session_admissions WHERE principal_id='api'
            AND target_session_id=? AND payload_digest=? AND request_id=?''',
            (ref.session_id, digest, settings['room_input_media']['request_id'])).fetchone()
        if admission is None:
            raise RuntimeStoreError('permission_denied')
        copies = iter(admission_input_refs(conn, admission) or ())
        documents, images = [], []
        for item, reference in zip(manifest, references):
            if item['kind'] == 'image':
                images.extend(restore_native_media([reference]))
                continue
            retained = next(copies, None)
            if retained is None:
                raise RuntimeStoreError('storage_unavailable')
            path = copy_path(authority.db, retained)
            if reference != {'path': str(path), 'sha256': retained['digest'], 'size': retained['size']}:
                raise RuntimeStoreError('permission_denied')
            if verified_identity(path, retained['digest'], retained['size']) != (retained['device'], retained['inode']):
                raise RuntimeStoreError('storage_unavailable')
            documents.append(f"- {item['name']}: {path}")
        if next(copies, None) is not None:
            raise RuntimeStoreError('permission_denied')
    if documents:
        text += '\n\nUse the file tools to inspect these shared files:\n' + '\n'.join(documents)
    if images:
        from agent.image_routing import build_native_content_parts
        content, skipped = build_native_content_parts(text, images)
        if skipped:
            raise RuntimeStoreError('storage_unavailable')
        return content
    return text
