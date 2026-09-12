"""Private Files working copies backing the existing native document wire paths.

Custody is established by the authorized Files materializer, never by inspecting
prompt text. Both publication and native GC serialize on the same profile DB.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile

from gateway.hosted_room_attachments import _name, default_attachment_root
from hermes_state_runtime import RuntimeStoreError, _epoch, _json

_WITNESS = 'gateway.hosted.input-custody.v1:'
_IDENTITY = ('principal_id', 'target_session_id', 'request_id', 'payload_digest')


def _root(db_path):
    return default_attachment_root(Path(db_path).resolve()) / 'working-documents'


def _private_directory(path):
    if path.resolve() != path:
        raise RuntimeStoreError('storage_unavailable')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.resolve() != path or not path.is_dir():
        raise RuntimeStoreError('storage_unavailable')


def _verify(path, size, digest):
    from gateway.session_ingress_media import _open_regular
    try:
        with _open_regular(path) as source:
            if os.fstat(source.fileno()).st_size != size or hashlib.file_digest(source, 'sha256').hexdigest() != digest:
                raise ValueError('changed working copy')
    except (OSError, ValueError) as exc:
        raise RuntimeStoreError('storage_unavailable') from exc


def _copy(data, target, digest):
    """Publish only complete private bytes; never repair an existing changed copy."""
    from gateway.session_ingress_media import _sync_directory
    if target.exists() or target.is_symlink():
        _verify(target, len(data), digest)
        return
    fd, temporary = tempfile.mkstemp(prefix='.document-', dir=target.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        _sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def retain_document(store, name, data):
    """Retain verified Files bytes while keeping old admission/retry paths exact."""
    from gateway.session_ingress_media import _media_root, _sync_directory, restore_native_media
    if _name(name) != name:
        raise RuntimeStoreError('invalid_params')
    digest = hashlib.sha256(data).hexdigest()
    root = _root(store.db_path)
    backing = root / digest / name
    native_root = _media_root()
    alias = native_root / digest / name
    reference = {'path': str(alias), 'sha256': digest, 'size': len(data)}
    # Store reads/owner snapshots already authorized these bytes. No canonical
    # source blob is linked: only these two private working copies may share an inode.
    with store._lock, store._transaction(immediate=True):
        for directory in (root, backing.parent, native_root, alias.parent):
            _private_directory(directory)
        _copy(data, backing, digest)
        if not alias.exists() and not alias.is_symlink():
            try:
                os.link(backing, alias)
            except OSError as exc:
                if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOSYS,
                                     errno.EOPNOTSUPP, errno.ENOTSUP}:
                    raise
                _copy(data, alias, digest)
        # An accepted alias may already exist. Verify it; never overwrite it.
        restore_native_media([reference])
        for directory in (alias.parent, native_root, native_root.parent, backing.parent, root, root.parent):
            _sync_directory(directory)
    return reference


def holds_native_reference(db_path, reference):
    """Called under the GC writer lock; uncertain private custody means retain."""
    root = _root(db_path)
    if root.resolve() != root:
        return True
    backing = root / reference['sha256'] / Path(reference['path']).name
    # Corrupt/symlinked copies must not become implicit deletion authority.
    # Materialization still verifies every byte before it can serve the input.
    return backing.exists() or backing.is_symlink()


def legacy_custody_missing(conn):
    """No timestamp/cutoff guess: every unwitnessed retained hosted row may own bytes."""
    rows = conn.execute('''SELECT a.principal_id,a.target_session_id,a.request_id,a.payload_digest,m.value
        FROM session_admissions a LEFT JOIN state_meta m ON m.key=? || a.admission_id
        WHERE a.request_id LIKE 'hosted:%' ''', (_WITNESS,))
    for row in rows:
        try:
            if json.loads(row[4]) != {'version': 1, **dict(zip(_IDENTITY, row[:4]))}:
                return True
        except (TypeError, ValueError):
            return True
    return False


def record_admission_custody(rpc, request_id, admission_id):
    """Private evidence after authorized materialization and exact canonical admission.

    Missing evidence is safe (GC retains conservatively), so callers can preserve
    an accepted receipt if this bookkeeping cannot be committed.
    """
    def write(conn):
        _epoch(conn, rpc.authority.epoch)
        row = conn.execute('''SELECT principal_id,target_session_id,request_id,payload_digest
            FROM session_admissions WHERE admission_id=?''', (admission_id,)).fetchone()
        if row is None:
            return  # Already retired to terminal-only evidence; never reactivate it.
        if tuple(row[:3]) != (rpc.principal.subject, rpc.ref.session_id, request_id):
            raise RuntimeStoreError('permission_denied')
        proof = _json({'version': 1, **dict(zip(_IDENTITY, row))})
        key = _WITNESS + admission_id
        old = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        if old is not None and old[0] != proof:
            raise RuntimeStoreError('storage_unavailable')
        conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)', (key, proof))
    rpc.authority.db._execute_write(write)
