"""Read retained original input authority without recapture or byte-store setup."""
import json

from gateway.hosted_room_driver import validate_bound_task_manifest
from gateway.hosted_room_input_reclamation import copy_path, verified_identity
from gateway.session_ingress_media import _ATTACHMENT_MIMES
from hermes_state_input_custody import admission_input_refs, IDENTITY_FIELDS
from hermes_state_runtime import RuntimeStoreError


def retained_hosted_input(conn, db, *, room_id, member_id, task_payload, admission):
    """Called inside the stopped owner's writer; missing evidence is not emptiness.

    The source's committed event/recipient/digest metadata and admission-owned v3
    references already persist the binding. Expired source bytes are not read.
    Original document custody must still be ready; this does not release it.
    """
    manifest = validate_bound_task_manifest(task_payload['attachments'])
    stored = json.loads(admission['payload_json'])
    refs = admission_input_refs(conn, admission) or []
    documents = [item for item in manifest if item['mime'] not in _ATTACHMENT_MIMES]
    if len(refs) != len(documents):
        raise RuntimeStoreError('input_binding_unavailable')
    preparations = []
    if documents:
        preparations = [dict(r) for r in conn.execute(
            "SELECT preparation_id,owner_epoch,principal_id,target_session_id,request_id,intent,payload_digest,admission_id "
            "FROM input_custody_preparations WHERE admission_id=? AND state='consumed'",
            (admission['admission_id'],))]
        if len(preparations) != 1 or preparations[0]['owner_epoch'] != admission['owner_epoch'] or any(
                preparations[0][k] != admission[k] for k in IDENTITY_FIELDS):
            raise RuntimeStoreError('input_binding_unavailable')
    sources, paths, images, types = [], [], [], []
    media = (stored.get('attachments_v1') or {}).get('media', [])
    for item in manifest:
        source = conn.execute('SELECT * FROM hosted_room_attachments WHERE attachment_id=? AND room_id=?',
                              (item['attachment_id'], room_id)).fetchone()
        if (source is None or source['state'] != 'committed' or any(source[k] != item[k]
                for k in ('event_id', 'name', 'kind', 'mime', 'size'))
                or member_id not in json.loads(source['recipient_member_ids_json'])):
            raise RuntimeStoreError('input_binding_unavailable')
        event = conn.execute('SELECT payload_json FROM hosted_room_events WHERE room_id=? AND event_id=?',
                             (room_id, item['event_id'])).fetchone()
        original = {k: item[k] for k in ('attachment_id', 'name', 'kind', 'mime', 'size')}
        if event is None or original not in json.loads(event[0]).get('attachments', []):
            raise RuntimeStoreError('input_binding_unavailable')
        sources.append({k: source[k] for k in ('attachment_id', 'room_id', 'event_id', 'name', 'kind',
                                               'mime', 'size', 'sha256', 'recipient_member_ids_json')})
        if item['mime'] in _ATTACHMENT_MIMES:
            index = len(images)
            if index >= len(media) or (media[index]['sha256'], media[index]['size']) != (source['sha256'], item['size']):
                raise RuntimeStoreError('input_binding_unavailable')
            images.append(media[index])  # Original admitted reference, never a guessed path.
            types.append(item['mime'])
        else:
            copy = refs[len(paths)]
            if (copy['name'], copy['digest'], copy['size']) != (item['name'], source['sha256'], item['size']):
                raise RuntimeStoreError('input_binding_unavailable')
            path = copy_path(db, copy)
            if verified_identity(path, copy['digest'], copy['size']) != (copy['device'], copy['inode']):
                raise RuntimeStoreError('input_binding_unavailable')
            paths.append(str(path))
    expected = {'text': task_payload['prompt'] + ''.join('\n[Shared attachment] file: ' + p + '\n' for p in paths)}
    if images:
        expected['attachments_v1'] = {'media': images, 'media_types': types}
    if 'local_operator_v1' in stored:
        expected['local_operator_v1'] = stored['local_operator_v1']
    if expected != stored:
        raise RuntimeStoreError('input_binding_unavailable')
    # Native image custody has its own exact admitted-reference verifier. Never
    # go back to source attachments or manufacture a new native capture.
    from gateway.session_ingress_media import restore_native_media
    restore_native_media(images)
    return dict(sources=sources, copies=refs, preparations=preparations,
                payload_digest=admission['payload_digest'], payload=expected)
