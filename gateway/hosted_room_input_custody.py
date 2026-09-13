"""Bounded legacy inventory and pre-admission private document custody."""
import hashlib
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile

from gateway.hosted_room_attachments import default_attachment_root
from hermes_state_runtime import RuntimeStoreError, _json
from hermes_state_terminal import terminal_admission

_READY = 'gateway.input-custody.v2'
_FIELDS = ('principal_id', 'target_session_id', 'request_id', 'payload_digest')
_DIGEST = re.compile(r'[0-9a-f]{64}')
_DECIMAL = re.compile(r'0|[1-9][0-9]*')
_IDENTITY_FORMAT = 'device-inode-decimal-v1'


def _stored_identity(device, inode):
    if any(not isinstance(value, str) or not _DECIMAL.fullmatch(value) for value in (device, inode)):
        raise ValueError('invalid saved file identity')
    result = int(device), int(inode)
    if result[1] == 0:
        raise ValueError('unknown saved inode')
    return result


def _ready(conn, root):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_READY,)).fetchone()
    if row is None or row[0] != _json({'version': 2, 'root': str(root), 'identity_format': _IDENTITY_FORMAT}):
        return False
    try:
        columns = {row[1]: (row[2].upper(), row[3])
                   for row in conn.execute('PRAGMA table_info(gateway_legacy_input_paths)')}
        if any(columns.get(field) != ('TEXT', 1) for field in ('device', 'inode')):
            return False
        for device, inode in conn.execute('SELECT device,inode FROM gateway_legacy_input_paths'):
            _stored_identity(device, inode)
    except (sqlite3.Error, ValueError, TypeError):
        return False
    return True


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
            conn.execute('SELECT path,digest,device,inode FROM gateway_legacy_input_paths LIMIT 0')
            conn.execute('SELECT admission_id,principal_id,target_session_id,request_id,payload_digest FROM gateway_legacy_input_admissions LIMIT 0')
            return
        conn.execute('''CREATE TABLE IF NOT EXISTS gateway_legacy_input_paths(
            path TEXT PRIMARY KEY, digest TEXT NOT NULL,
            device TEXT NOT NULL, inode TEXT NOT NULL)''')
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
                    observed = path.stat(follow_symlinks=False)
                    if not stat.S_ISREG(observed.st_mode):
                        raise RuntimeStoreError('storage_unavailable')
                    device, inode = str(observed.st_dev), str(observed.st_ino)
                    _stored_identity(device, inode)
                    conn.execute('INSERT INTO gateway_legacy_input_paths VALUES(?,?,?,?)',
                                 (str(path.relative_to(root)), directory.name, device, inode))
        conn.executemany('INSERT INTO gateway_legacy_input_admissions VALUES(?,?,?,?,?)', rows)
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (_READY, _json({'version': 2, 'root': str(root), 'identity_format': _IDENTITY_FORMAT})))
        if not _ready(conn, root):
            raise RuntimeStoreError('storage_unavailable')
    try:
        db._execute_write(initialize)
    except (OSError, ValueError, sqlite3.Error) as exc:
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
    for row in conn.execute('SELECT path,device,inode FROM gateway_legacy_input_paths WHERE digest=?', (reference['sha256'],)):
        old = root / row[0]
        if old == path:
            return True
        try:
            candidate = _file_identity(path)
            if _stored_identity(row[1], row[2]) == candidate or _file_identity(old) == candidate:
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
