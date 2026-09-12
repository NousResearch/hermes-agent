"""Retain signed peer inputs using the canonical runtime's immutable media store."""
import os
from pathlib import Path
import tempfile

from gateway.hosted_room_peer import canonical_attachment_manifest, attachment_manifest_digest
from gateway.session_ingress_media import capture_native_media, restore_native_media, validate_media_batch_size
from hermes_state_runtime import RuntimeStoreError


def peer_input_available(adapter):
    from gateway.session_authorities import active_authority
    runner = getattr(adapter, 'gateway_runner', None)
    authority = active_authority(runner) if runner is not None else None
    return authority is not None and adapter._ensure_session_db() is authority.db


def retain_peer_input(spool, dispatch):
    items = spool.materialize(dispatch)
    manifest = canonical_attachment_manifest([
        {key: value for key, value in item.items() if key != 'path'} for item in items])
    if attachment_manifest_digest(manifest) != dispatch.attachment_manifest_digest:
        raise RuntimeStoreError('permission_denied')
    validate_media_batch_size(item['size'] for item in manifest)
    references = []
    for item in items:
        data = spool._read_verified(Path(item['path']), size=item['size'], digest=item['sha256'])
        with tempfile.TemporaryDirectory(prefix='hermes-peer-input-') as directory:
            path = Path(directory) / item['name']
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'wb') as output:
                output.write(data)
            reference = capture_native_media([path])[0]
        if (reference['size'], reference['sha256']) != (item['size'], item['sha256']):
            raise RuntimeStoreError('storage_unavailable')
        references.append(reference)
    return {'manifest': manifest, 'media': references}


def check_peer_input(settings):
    data = settings.get('room_input_media')
    dispatch = settings.get('room_dispatch') or {}
    if data is None:
        if dispatch.get('attachment_manifest_digest') is not None:
            raise RuntimeStoreError('storage_unavailable')
        return [], []
    if not isinstance(data, dict) or set(data) != {'manifest', 'media'}:
        raise RuntimeStoreError('invalid_params')
    manifest = canonical_attachment_manifest(data['manifest'])
    references = data['media']
    if (not isinstance(references, list) or len(references) != len(manifest)
            or attachment_manifest_digest(manifest) != dispatch.get('attachment_manifest_digest')):
        raise RuntimeStoreError('permission_denied')
    if any(not isinstance(reference, dict) or Path(reference.get('path', '')).name != item['name']
           for reference, item in zip(references, manifest)):
        raise RuntimeStoreError('permission_denied')
    paths = restore_native_media(references)
    if any((reference['size'], reference['sha256']) != (item['size'], item['sha256'])
           for reference, item in zip(references, manifest)):
        raise RuntimeStoreError('permission_denied')
    return manifest, paths


def peer_input_content(text, settings):
    manifest, paths = check_peer_input(settings)
    if not manifest:
        return text
    images = [path for item, path in zip(manifest, paths) if item['kind'] == 'image']
    documents = [f"- {item['name']}: {path}" for item, path in zip(manifest, paths) if item['kind'] != 'image']
    if documents:
        text += '\n\nUse the file tools to inspect these shared files:\n' + '\n'.join(documents)
    if images:
        from agent.image_routing import build_native_content_parts
        content, skipped = build_native_content_parts(text, images)
        if skipped:
            raise RuntimeStoreError('storage_unavailable')
        return content
    return text
