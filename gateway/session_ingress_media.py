"""Immutable local-media references for owner-only native admission.

The existing managed document cache supplies profile routing, delivery eligibility
and upload limits. Its flat age-based cleanup skips this retained subdirectory:
native bytes are released by ``release_admission_media`` once their row is
terminal and no live native input or retained API image context still holds them.
"""
import hashlib
import json
import os
from pathlib import Path
import re
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
_IMAGE_EXT = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/gif': '.gif', 'image/webp': '.webp'}


def sniff_image_mime(data):
    """The admission image type the bytes really are (png/jpeg/gif/webp), else ``None``."""
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return None


def admit_attachments(attachments, *, admitted=None):
    """Wire ``attachments: [{path, mime}]`` -> committed payload fields (``{}`` when absent).

    ``admitted()`` returns the durable admission already holding this request's identity (or
    ``None``): a lost-ACK retry whose disposable staging file is gone reconciles against it."""
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
    mimes = [item['mime'] for item in attachments]
    row = admitted() if admitted is not None else None
    committed = row['payload'].get('attachments_v1') if row is not None else None
    try:
        hashes = _attachment_hashes(paths)
    except (OSError, ValueError) as exc:
        raise RuntimeStoreError('invalid_params') from exc
    if None in hashes:
        if (committed is None or [Path(r['path']).name for r in committed['media']] != [p.name for p in paths]
                or committed['media_types'] != mimes):
            raise RuntimeStoreError('invalid_params')
        if any(digest is not None and digest != reference['sha256']
               for digest, reference in zip(hashes, committed['media'])):
            raise RuntimeStoreError('admission_conflict')
        retry = True
    else:
        # Retry identity is the committed bytes, not the disposable staging name.
        retry = (committed is not None and committed['media_types'] == mimes
                 and [r['sha256'] for r in committed['media']] == hashes)
    if retry:
        # A live row still executes these bytes, so they must verify; a terminal row is
        # exact-retry evidence by digest only. ``admit_session_input`` still checks the digest.
        if row['status'] != 'terminal':
            restore_native_media(committed['media'])
        return {'attachments_v1': committed}
    return {'attachments_v1': {'media': capture_native_media(paths), 'media_types': mimes}}


def _attachment_hashes(paths):
    hashes = []
    for path in paths:
        try:
            # Staging must not provide a second name for a file outside it.
            if path.lstat().st_nlink != 1:
                raise RuntimeStoreError('invalid_params')
            hashes.append(_sha256(path))
        except FileNotFoundError:
            # One pruned file cannot skip validation of the remaining batch.
            hashes.append(None)
    return hashes


def _sha256(path):
    with _open_regular(path) as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


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
    # Only private snapshots belong exclusively to this batch. Published aliases
    # can be reused by another capture before either admission is committed.
    staged = []
    try:
        for value in paths:
            _capture_file(Path(value), limit, total, references, staged)
            total = sum(reference['size'] for reference in references)
        for temporary, reference in zip(staged, references):
            target = Path(reference['path'])
            root = target.parent.parent
            if target.parent.resolve() != target.parent:
                raise RuntimeStoreError('invalid_params')
            target.parent.mkdir(mode=0o700, exist_ok=True)
            if target.exists() or target.is_symlink():
                restore_native_media([reference])
            else:
                os.replace(temporary, target)
            for directory in (target.parent, root, root.parent, root.parent.parent, root.parent.parent.parent):
                _sync_directory(directory)
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    return references


def _capture_file(path, limit, total, references, staged):
    from gateway.platforms.base import validate_inbound_media_size
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
        staged.append(temporary)
        with os.fdopen(fd, 'wb') as output:
            digest, size = hashlib.sha256(), 0
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
        reference = {'path': str(target), 'sha256': digest.hexdigest(), 'size': size}
        if target.exists() or target.is_symlink():
            restore_native_media([reference])
        references.append(reference)


def validate_media_batch_size(sizes):
    """Preflight validated manifest sizes before materializing any batch member."""
    from gateway.platforms.base import get_inbound_media_max_bytes, validate_inbound_media_size
    try:
        validate_inbound_media_size(sum(sizes), max_bytes=max(0, get_inbound_media_max_bytes()))
    except ValueError as exc:
        raise RuntimeStoreError('invalid_params') from exc


def admission_media_references(payload):
    """Native references eligible as deletion candidates after terminal settlement."""
    return list(payload.get('attachments_v1', {}).get('media', ())) + list(
        payload.get('native_text_v1', {}).get('media', ()))


def _held_media_paths(conn):
    # Project only references, not potentially large inline-image/history payloads.
    rows = conn.execute("""SELECT status, json_extract(payload_json,
            '$.attachments_v1.media', '$.native_text_v1.media', '$.api_turn_v1.media')
            FROM session_admissions WHERE status!='terminal'
            OR json_type(payload_json, '$.api_turn_v1.media') IS NOT NULL""").fetchall()
    held = set()
    for status, encoded in rows:
        attachments, native, api = json.loads(encoded)
        # API images remain canonical history context after the turn completes.
        references = list(api or ())
        if status != 'terminal':
            references.extend(attachments or ())
            references.extend(native or ())
        held.update(reference['path'] for reference in references)
    return held


def _file_identity(path):
    saved = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(saved.st_mode) or not saved.st_ino:
        raise ValueError('uncertain native file identity')
    return saved.st_dev, saved.st_ino


def _held_file_identities(paths, root):
    identities = set()
    try:
        for value in paths:
            path = Path(value)
            if path.parent.parent != root or path.parent.resolve() != path.parent:
                return None
            identities.add(_file_identity(path))
    except (OSError, ValueError):
        return None
    return identities


def release_admission_media(db, admission_id):
    """Delete eligible terminal native bytes unless another retained input holds them.

    Terminal rows are exact-retry evidence by digest only; their bytes are not
    replayed. Rows that are not terminal (queued, started, unknown) may still
    execute, so any path they reference stays on disk; API image references stay
    on disk in every status because they remain history context after settlement,
    and are holders only (see ``collect_unheld_api_images``). Different regular
    files can collect independently; physical aliases and uncertain stat results
    retain conservatively. A shared hardlink can consequently retain an extra old alias.
    """
    from hermes_state_runtime import get_session_admission
    row = get_session_admission(db, admission_id=admission_id)
    if row is None or row['status'] != 'terminal':
        return 0
    return release_unheld_media(db, admission_media_references(row['payload']), retain_history=False)


_API_IMAGE_NAME = re.compile(r'api_[0-9a-f]{32}\.(png|jpg|gif|webp)')


def collect_unheld_api_images(db):
    """Collect retained API images no admission holds any more: deleting a chat retires its
    admissions and erases their references, so those bytes have no owner left. Only the exact
    name ``commit_api_images`` gives (``api_<sha256[:32]><ext>`` under its own digest) is a
    candidate; retained hosted documents are held by prompt text, never by a media reference."""
    root = _media_root()
    return release_unheld_media(db, [
        {'path': str(path), 'sha256': path.parent.name} for path in (root.glob('*/api_*') if root.is_dir() else ())
        if _API_IMAGE_NAME.fullmatch(path.name) and path.name[4:36] == path.parent.name[:32]])


def release_unheld_media(db, references, *, retain_history=True):
    """Delete unowned references, preserving accepted admission and retained transcript owners."""
    if not references:
        return 0
    root = _media_root()
    def collect(conn):
        held = _held_media_paths(conn)
        identities = _held_file_identities(held, root)
        if identities is None:
            return 0
        released = 0
        for reference in references:
            path = Path(reference['path'])
            if reference['path'] in held or path.parent.parent != root or path.parent.name != reference['sha256']:
                continue
            if retain_history and conn.execute("SELECT 1 FROM messages WHERE instr(content,?) OR instr(content,?) LIMIT 1",
                    (reference['path'], json.dumps(reference['path'])[1:-1])).fetchone():
                continue
            try:
                if path.parent.resolve() != path.parent or _file_identity(path) in identities:
                    continue
                path.unlink()
                released += 1
                path.parent.rmdir()
            except (OSError, ValueError):
                continue
        return released
    return db._execute_write(collect)


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
