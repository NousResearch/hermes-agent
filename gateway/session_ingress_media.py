"""Immutable local-media references for owner-only native admission.

The existing managed document cache supplies profile routing, delivery eligibility
and upload limits. Its flat age-based cleanup skips this retained subdirectory:
retained bytes are released per admission by ``release_admission_media`` once the
row is terminal and no live (queued/started/unknown) row still references them.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from hermes_state_runtime import RuntimeStoreError


def _media_root():
    from gateway.platforms.base import get_document_cache_dir
    return get_document_cache_dir().resolve() / 'native-inputs'


# Public ``prompt.submit`` attachments: a local client stages image bytes in the
# profile image cache (where messaging adapters stage downloads), the authority
# commits them as immutable bytes at admission so no later mutation or cache
# cleanup of the staging file can change what executes.
_ATTACHMENT_MIMES = frozenset({'image/png', 'image/jpeg', 'image/gif', 'image/webp'})
_ATTACHMENT_LIMIT = 10


def admit_attachments(attachments):
    """Wire ``attachments: [{path, mime}]`` -> committed payload fields (``{}`` when absent)."""
    if attachments is None:
        return {}
    if (not isinstance(attachments, list) or not attachments or len(attachments) > _ATTACHMENT_LIMIT
            or any(not isinstance(item, dict) or set(item) != {'path', 'mime'}
                   or not isinstance(item['path'], str) or item['mime'] not in _ATTACHMENT_MIMES
                   for item in attachments)):
        raise RuntimeStoreError('invalid_params')
    from gateway.platforms.base import get_image_cache_dir
    staging = get_image_cache_dir().resolve()
    paths = [Path(item['path']) for item in attachments]
    if any(not path.is_absolute() or path.resolve().parent != staging for path in paths):
        raise RuntimeStoreError('invalid_params')
    return {'attachments_v1': {'media': capture_native_media(paths),
                               'media_types': [item['mime'] for item in attachments]}}


def restore_attachments(payload):
    """Committed attachment fields -> ``MessageEvent`` media kwargs (``{}`` for text-only rows)."""
    data = payload.get('attachments_v1')
    if not data:
        return {}
    return {'media_urls': restore_native_media(data['media']), 'media_types': list(data['media_types'])}


def _sync_directory(path):
    # Windows cannot open directories through os.open; file fsync still precedes ACK.
    if os.name != 'nt':
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _open_regular(path):
    if not path.is_absolute() or path.is_symlink():
        raise ValueError('native media must be a local regular file')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0))
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError('native media must be a regular file')
    return os.fdopen(fd, 'rb')


def capture_native_media(paths):
    from gateway.platforms.base import get_inbound_media_max_bytes, validate_inbound_media_size
    references = []
    limit = max(0, get_inbound_media_max_bytes())
    # ``gateway.max_inbound_media_bytes`` bounds the whole admission, not each file: with
    # per-file caps alone ten attachments could commit ~1.25 GiB of retained bytes per turn.
    total = 0
    for value in paths:
        path = Path(value)
        try:
            source = _open_regular(path)
        except (OSError, ValueError) as exc:
            raise RuntimeStoreError('invalid_params') from exc
        with source:
            root = _media_root()
            if root.resolve() != root:
                raise RuntimeStoreError('invalid_params')
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix='.capture-', dir=root)
            temporary = Path(name)
            try:
                digest, size = hashlib.sha256(), 0
                with os.fdopen(fd, 'wb') as output:
                    while chunk := source.read(1024 * 1024):
                        size += len(chunk)
                        try:
                            validate_inbound_media_size(total + size, max_bytes=limit)
                        except ValueError as exc:
                            raise RuntimeStoreError('invalid_params') from exc
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                target = root / digest.hexdigest() / path.name
                if target.parent.resolve() != target.parent:
                    raise RuntimeStoreError('invalid_params')
                target.parent.mkdir(mode=0o700, exist_ok=True)
                reference = {'path': str(target), 'sha256': digest.hexdigest(), 'size': size}
                if target.exists():
                    # An earlier admission can reference this file. Never repair it by
                    # overwriting accepted bytes, even when a retry has the same digest.
                    restore_native_media([reference])
                else:
                    os.replace(temporary, target)
                for directory in (target.parent, root, root.parent, root.parent.parent, root.parent.parent.parent):
                    _sync_directory(directory)
                references.append(reference)
                total += size
            finally:
                temporary.unlink(missing_ok=True)
    return references


def admission_media_references(payload):
    """Every retained ``native-inputs`` reference a committed payload owns."""
    return list(payload.get('attachments_v1', {}).get('media', ())) + list(
        payload.get('native_text_v1', {}).get('media', ()))


def release_admission_media(db, admission_id):
    """Delete retained bytes of a terminal admission unless a live row still shares them.

    Terminal rows are exact-retry evidence by digest only; their bytes are not
    replayed. Rows that are not terminal (queued, started, unknown) may still
    execute, so any digest they reference stays on disk.
    """
    from hermes_state_runtime import get_session_admission
    row = get_session_admission(db, admission_id=admission_id)
    if row is None or row['status'] != 'terminal':
        return 0
    mine = admission_media_references(row['payload'])
    if not mine:
        return 0
    root = _media_root()
    with db._read_ctx() as conn:
        live = conn.execute("SELECT payload_json FROM session_admissions WHERE status!='terminal'").fetchall()
    held = {reference['sha256'] for saved in live
            for reference in admission_media_references(json.loads(saved[0]))}
    released = 0
    for reference in mine:
        path = Path(reference['path'])
        if reference['sha256'] in held or path.parent.parent != root or path.parent.name != reference['sha256']:
            continue
        try:
            path.unlink()
            released += 1
            path.parent.rmdir()
        except OSError:
            continue
    return released


def restore_native_media(references):
    if not references:
        return []
    root = _media_root()
    paths = []
    try:
        for reference in references:
            if set(reference) != {'path', 'sha256', 'size'}:
                raise ValueError('invalid media reference')
            path = Path(reference['path'])
            if (path.parent.parent != root or path.resolve() != path
                    or path.parent.name != reference['sha256']):
                raise ValueError('foreign media reference')
            with _open_regular(path) as source:
                if os.fstat(source.fileno()).st_size != reference['size']:
                    raise ValueError('changed media size')
                if hashlib.file_digest(source, 'sha256').hexdigest() != reference['sha256']:
                    raise ValueError('changed media bytes')
            paths.append(str(path))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeStoreError('storage_unavailable') from exc
    return paths
