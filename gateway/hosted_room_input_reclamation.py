"""Explicit owner-scoped initialization and bounded sealed private-copy cleanup."""
import hashlib
import os
from pathlib import Path
import re
import stat
import time
import uuid

from gateway.hosted_room_attachments import default_attachment_root
from gateway.hosted_room_input_custody import _ready, _legacy_active, _stored_identity
from gateway.session_ingress_media import _media_root, _file_identity, _held_media_paths, _held_file_identities, _sync_directory
from hermes_state_input_custody import READY_KEY, create_schema, copy_is_held, seal_copy
from hermes_state_runtime import RuntimeStoreError, _epoch, _json

_DIGEST = re.compile(r'[0-9a-f]{64}')


def owned_home(db):
    from gateway.runtime_ownership import process_ownership
    from hermes_constants import get_hermes_home
    home = Path(db.db_path).resolve().parent
    if get_hermes_home().resolve() != home or not process_ownership.owns(home):
        raise RuntimeStoreError('permission_denied')
    return home


def copy_path(db, copy):
    root = _media_root() if copy['namespace'] == 'alias' else default_attachment_root(db.db_path) / (
        'working-documents-v3' if copy['namespace'] == 'v3' else 'working-documents-v2')
    if copy['namespace'] not in {'v3', 'v2', 'alias'} or not _DIGEST.fullmatch(copy['digest']):
        raise RuntimeStoreError('storage_unavailable')
    name = copy['name']
    if not name or Path(name).name != name or name in {'.', '..'}:
        raise RuntimeStoreError('storage_unavailable')
    path = root.resolve() / copy['digest'] / name
    if root.resolve() != root or path.parent.resolve() != path.parent:
        raise RuntimeStoreError('storage_unavailable')
    return path


def require_initialized(conn, db):
    marker = conn.execute('SELECT value FROM state_meta WHERE key=?', (READY_KEY,)).fetchone()
    if not _ready(conn, _media_root()) or marker is None or marker[0] != _json({'version': 3, 'home': str(Path(db.db_path).resolve().parent)}):
        raise RuntimeStoreError('storage_unavailable')


def verified_identity(path, digest, size):
    from gateway.session_ingress_media import _open_regular
    if path.resolve() != path:
        raise RuntimeStoreError('storage_unavailable')
    try:
        with _open_regular(path) as source:
            saved = os.fstat(source.fileno())
            if (saved.st_size != size or not saved.st_ino
                    or hashlib.file_digest(source, 'sha256').hexdigest() != digest):
                raise ValueError('private copy changed')
            return str(saved.st_dev), str(saved.st_ino)
    except (OSError, ValueError) as exc:
        raise RuntimeStoreError('storage_unavailable') from exc


def initialize_working_copies(db, *, epoch):
    """One-time public-v2 inventory; parent invokes only before owner ingress."""
    home = owned_home(db)
    def initialize(conn):
        _epoch(conn, epoch)
        if not _ready(conn, _media_root()):
            raise RuntimeStoreError('storage_unavailable')
        marker = conn.execute('SELECT value FROM state_meta WHERE key=?', (READY_KEY,)).fetchone()
        if marker is not None:
            require_initialized(conn, db)
            conn.execute('SELECT copy_id,state,generation FROM input_custody_copies LIMIT 0')
            return
        create_schema(conn)
        if conn.execute('SELECT 1 FROM input_custody_copies LIMIT 1').fetchone():
            raise RuntimeStoreError('storage_unavailable')
        root = default_attachment_root(db.db_path) / 'working-documents-v2'
        if root.resolve() != root:
            raise RuntimeStoreError('storage_unavailable')
        if root.exists():
            for directory in root.iterdir():
                if not _DIGEST.fullmatch(directory.name):
                    continue
                if not stat.S_ISDIR(directory.stat(follow_symlinks=False).st_mode):
                    raise RuntimeStoreError('storage_unavailable')
                for path in directory.iterdir():
                    saved = path.stat(follow_symlinks=False)
                    if not stat.S_ISREG(saved.st_mode):
                        raise RuntimeStoreError('storage_unavailable')
                    identity = verified_identity(path, directory.name, saved.st_size)
                    conn.execute('INSERT INTO input_custody_copies VALUES(?,?,?,?,?,?,?,?,?)',
                        (uuid.uuid4().hex, 'v2', path.name, directory.name, saved.st_size, 1, 'ready', *identity))
                    alias = _media_root() / directory.name / path.name
                    if alias.exists() or alias.is_symlink():
                        identity = verified_identity(alias, directory.name, saved.st_size)
                        conn.execute('INSERT OR IGNORE INTO input_custody_copies VALUES(?,?,?,?,?,?,?,?,?)',
                            (uuid.uuid4().hex, 'alias', path.name, directory.name, saved.st_size, 1, 'ready', *identity))
        conn.execute('''INSERT INTO input_custody_legacy_ids
            SELECT admission_id,principal_id,target_session_id,request_id,payload_digest,intent
            FROM session_admissions WHERE request_id LIKE 'hosted:%' ''')
        # Pre-lease branches may have copied opaque old input paths. Bound the
        # hold to these existing branches, not arbitrary future conversations.
        conn.execute('''INSERT INTO input_custody_legacy_branches
            SELECT id FROM sessions WHERE json_extract(model_config,'$._branched_from') IS NOT NULL''')
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (READY_KEY, _json({'version': 3, 'home': str(home)})))
    db._execute_write(initialize)


def _legacy_holds(conn, path, copy):
    if not _legacy_active(conn):
        return False
    identity = _file_identity(path)
    for row in conn.execute('SELECT path,device,inode FROM gateway_legacy_input_paths WHERE digest=?', (copy['digest'],)):
        old = _media_root() / row['path']
        if old == path or _stored_identity(row['device'], row['inode']) == identity:
            return True
        try:
            if _file_identity(old) == identity:
                return True
        except FileNotFoundError:
            continue
    return False


def _external_holds(conn, db, copy, path, now):
    held = _held_media_paths(conn)
    identities = _held_file_identities(held, _media_root())
    if identities is None or str(path) in held:
        return True
    try:
        identity = _file_identity(path)
        if copy['namespace'] == 'v3' and path.stat(follow_symlinks=False).st_nlink != 1:
            return True  # V3 never creates hardlinks; an extra owner is untracked.
    except FileNotFoundError:
        return False
    if identity in identities or _legacy_holds(conn, path, copy):
        return True
    # Other logical copies/preparations can refer to the same physical entry.
    for other in conn.execute('''SELECT * FROM input_custody_copies WHERE copy_id!=? AND state!='removed'
            AND ((device=? AND inode=?) OR (device IS NULL AND digest=?))''',
            (copy['copy_id'], str(identity[0]), str(identity[1]), copy['digest'])):
        if not copy_is_held(conn, other, now):
            continue
        other_path = copy_path(db, other)
        try:
            if other_path == path or _file_identity(other_path) == identity:
                return True
        except FileNotFoundError:
            if other['state'] == 'preparing':
                continue
            return True
    return False


def _collect(db, *, epoch, limit, aliases):
    owned_home(db)
    if type(limit) is not int or not 1 <= limit <= 256:
        raise RuntimeStoreError('invalid_params')
    now = time.time()
    def seal(conn):
        _epoch(conn, epoch)
        require_initialized(conn, db)
        cursor_key = READY_KEY + ('.alias-cursor' if aliases else '.copy-cursor')
        previous = conn.execute('SELECT value FROM state_meta WHERE key=?', (cursor_key,)).fetchone()
        cursor = previous[0] if previous else ''
        rows = conn.execute('''SELECT * FROM input_custody_copies WHERE state!='removed'
            AND (namespace='alias')=? AND copy_id>? ORDER BY copy_id LIMIT ?''', (int(aliases), cursor, limit)).fetchall()
        if not rows and cursor:
            rows = conn.execute('''SELECT * FROM input_custody_copies WHERE state!='removed'
                AND (namespace='alias')=? ORDER BY copy_id LIMIT ?''', (int(aliases), limit)).fetchall()
        conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)',
                     (cursor_key, rows[-1]['copy_id'] if rows else ''))
        selected = []
        for row in rows:
            copy = dict(row)
            if copy_is_held(conn, copy, now):
                continue
            path = copy_path(db, copy)
            try:
                if _external_holds(conn, db, copy, path, now):
                    continue
                if path.exists() or path.is_symlink():
                    identity = verified_identity(path, copy['digest'], copy['size'])
                    if copy['device'] is not None and identity != (copy['device'], copy['inode']):
                        continue
                    conn.execute('UPDATE input_custody_copies SET device=?,inode=? WHERE copy_id=?', (*identity, copy['copy_id']))
                if copy['state'] == 'sealed' or seal_copy(conn, copy, now):
                    selected.append((copy['copy_id'], copy['generation']))
            except (OSError, ValueError):
                continue  # Uncertain bytes/identity never authorize deletion.
        return selected, len(rows)
    selected, scanned = db._execute_write(seal)  # Seal MUST commit before the first unlink.
    removed = 0
    for copy_id, generation in selected:
        def unlink(conn):
            _epoch(conn, epoch)
            require_initialized(conn, db)
            row = conn.execute('SELECT * FROM input_custody_copies WHERE copy_id=?', (copy_id,)).fetchone()
            if row is None or row['state'] != 'sealed' or row['generation'] != generation:
                return 0
            path = copy_path(db, row)
            if copy_is_held(conn, row, time.time()) or _external_holds(conn, db, row, path, time.time()):
                return 0
            if path.exists() or path.is_symlink():
                if verified_identity(path, row['digest'], row['size']) != (row['device'], row['inode']):
                    return 0
                path.unlink()
                _sync_directory(path.parent)
            conn.execute("UPDATE input_custody_copies SET state='removed' WHERE copy_id=?", (copy_id,))
            return 1
        try:
            removed += db._execute_write(unlink)
        except (OSError, ValueError):
            continue  # The earlier committed seal survives; a later pass can finish.
    return {'scanned': scanned, 'sealed': len(selected), 'removed': removed}


def collect_working_copies(db, *, epoch, limit=64):
    """Online owner opportunity: v3 files and v2 backing, never old native aliases."""
    return _collect(db, epoch=epoch, limit=limit, aliases=False)


def collect_legacy_input_aliases(db, *, epoch, limit=64):
    """Parent calls ONLY in exclusive pre-ingress initialization, never housekeeping.

    Profile ownership/epoch is necessary, not proof of no in-flight captures.
    The bootstrap caller owns that startup-only contract.
    """
    return _collect(db, epoch=epoch, limit=limit, aliases=True)
