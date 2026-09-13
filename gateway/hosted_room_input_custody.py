"""Bounded legacy inventory and pre-admission private document custody."""
import errno
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile

from gateway.hosted_room_attachments import _name, default_attachment_root
from hermes_state_runtime import RuntimeStoreError, _json
from hermes_state_terminal import terminal_admission

_READY = 'gateway.input-custody.v2'
_FIELDS = ('principal_id', 'target_session_id', 'request_id', 'payload_digest')
_DIGEST = re.compile(r'[0-9a-f]{64}')


def _ready(conn, root):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_READY,)).fetchone()
    return row is not None and row[0] == _json({'version': 2, 'root': str(root)})


def initialize_input_custody(db):
    """After exclusive ownership, before recovery/traffic; failure refuses startup."""
    from gateway.session_ingress_media import _media_root
    root = _media_root()

    def initialize(conn):
        if root.resolve() != root:
            raise RuntimeStoreError('storage_unavailable')
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_READY,)).fetchone()
        if row is not None:
            if not _ready(conn, root):
                raise RuntimeStoreError('storage_unavailable')
            conn.execute('SELECT path,digest FROM gateway_legacy_input_paths LIMIT 0')
            conn.execute('SELECT admission_id,principal_id,target_session_id,request_id,payload_digest FROM gateway_legacy_input_admissions LIMIT 0')
            return
        conn.execute('''CREATE TABLE IF NOT EXISTS gateway_legacy_input_paths(
            path TEXT PRIMARY KEY, digest TEXT NOT NULL)''')
        conn.execute('CREATE INDEX IF NOT EXISTS gateway_legacy_input_digest ON gateway_legacy_input_paths(digest)')
        conn.execute('''CREATE TABLE IF NOT EXISTS gateway_legacy_input_admissions(
            admission_id TEXT PRIMARY KEY, principal_id TEXT NOT NULL,
            target_session_id TEXT NOT NULL, request_id TEXT NOT NULL, payload_digest TEXT NOT NULL)''')
        if (conn.execute('SELECT 1 FROM gateway_legacy_input_paths LIMIT 1').fetchone()
                or conn.execute('SELECT 1 FROM gateway_legacy_input_admissions LIMIT 1').fetchone()):
            raise RuntimeStoreError('storage_unavailable')
        rows = conn.execute('''SELECT admission_id,principal_id,target_session_id,request_id,payload_digest
            FROM session_admissions WHERE request_id LIKE 'hosted:%' ''').fetchall()
        if rows and root.exists():
            for directory in root.iterdir():
                if not _DIGEST.fullmatch(directory.name):
                    continue
                if not stat.S_ISDIR(directory.stat(follow_symlinks=False).st_mode):
                    raise RuntimeStoreError('storage_unavailable')
                for path in directory.iterdir():
                    if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                        raise RuntimeStoreError('storage_unavailable')
                    conn.execute('INSERT INTO gateway_legacy_input_paths VALUES(?,?)',
                                 (str(path.relative_to(root)), directory.name))
        conn.executemany('INSERT INTO gateway_legacy_input_admissions VALUES(?,?,?,?,?)', rows)
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (_READY, _json({'version': 2, 'root': str(root)})))
    try:
        db._execute_write(initialize)
    except (OSError, ValueError) as exc:
        raise RuntimeStoreError('storage_unavailable') from exc


def _legacy_active(conn):
    rows = conn.execute('SELECT * FROM gateway_legacy_input_admissions').fetchall()
    for row in rows:
        if conn.execute('SELECT 1 FROM session_admissions WHERE admission_id=?', (row[0],)).fetchone():
            return True
        try:
            retired = terminal_admission(conn, row[0])
            if (not isinstance(retired, dict) or retired.get('status') != 'terminal' or retired.get('admission_id') != row[0]
                    or any(retired.get(field) != value for field, value in zip(_FIELDS, row[1:]))):
                return True
        except (TypeError, ValueError):
            return True
    if rows:
        conn.execute('DELETE FROM gateway_legacy_input_paths')
        conn.execute('DELETE FROM gateway_legacy_input_admissions')
    return False


def _backing_root(db_path):
    return default_attachment_root(Path(db_path).resolve()) / 'working-documents-v2'


def custody_holds(conn, db_path, reference):
    """Under the GC writer lock. Missing initialization never enables deletion."""
    from gateway.session_ingress_media import _file_identity, _media_root
    root = _media_root()
    if not _ready(conn, root):
        return True
    path = Path(reference['path'])
    backing_root = _backing_root(db_path)
    backing = backing_root / reference['sha256'] / path.name
    try:
        backed = backing_root.resolve() != backing_root or backing.exists() or backing.is_symlink()
    except OSError:
        return True
    if backed:
        return True
    if not _legacy_active(conn):
        return False
    for row in conn.execute('SELECT path FROM gateway_legacy_input_paths WHERE digest=?', (reference['sha256'],)):
        old = root / row[0]
        if old == path:
            return True
        try:
            if _file_identity(old) == _file_identity(path):
                return True
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            return True
    return False


def _directory(path):
    if path.resolve() != path:
        raise RuntimeStoreError('storage_unavailable')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.resolve() != path:
        raise RuntimeStoreError('storage_unavailable')


def _verify(path, data):
    from gateway.session_ingress_media import _open_regular
    try:
        with _open_regular(path) as source:
            if os.fstat(source.fileno()).st_size != len(data) or hashlib.file_digest(source, 'sha256').digest() != hashlib.sha256(data).digest():
                raise ValueError('changed private copy')
    except (OSError, ValueError) as exc:
        raise RuntimeStoreError('storage_unavailable') from exc


def _copy(data, target):
    from gateway.session_ingress_media import _sync_directory
    if target.exists() or target.is_symlink():
        _verify(target, data)
        return
    fd, name = tempfile.mkstemp(prefix='.document-', dir=target.parent)
    temporary = Path(name)
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
    """Authorized bytes only; durable v2 backing before materializer handoff."""
    from gateway.session_ingress_media import _media_root, _sync_directory, restore_native_media
    if _name(name) != name:
        raise RuntimeStoreError('invalid_params')
    root = _media_root()
    digest = hashlib.sha256(data).hexdigest()
    backing_root = _backing_root(store.db_path)
    backing = backing_root / digest / name
    alias = root / digest / name
    reference = {'path': str(alias), 'sha256': digest, 'size': len(data)}
    with store._lock, store._transaction(immediate=True) as conn:
        if not _ready(conn, root):
            raise RuntimeStoreError('storage_unavailable')
        for directory in (backing_root, backing.parent, root, alias.parent):
            _directory(directory)
        _copy(data, backing)
        if not alias.exists() and not alias.is_symlink():
            try:
                os.link(backing, alias)
            except OSError as exc:
                if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP}:
                    raise
                _copy(data, alias)
        restore_native_media([reference])
        for directory in (alias.parent, root, root.parent, backing.parent, backing_root, backing_root.parent):
            _sync_directory(directory)
    return reference
