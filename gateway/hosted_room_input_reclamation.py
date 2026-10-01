"""Owner-scoped private input paths and bounded sealed-copy collection."""
import hashlib
import os
from pathlib import Path
import re
import time

from gateway.hosted_room_attachments import default_attachment_root
from gateway.session_ingress_media import _media_root, _file_identity, _held_media_paths, _held_file_identities, _sync_directory
from hermes_state_input_custody import copy_is_held, seal_copy
from hermes_state_runtime import RuntimeStoreError, _epoch

_DIGEST = re.compile(r'[0-9a-f]{64}')
_CURSORS = {'v3': 'gateway.input-reclamation.v3.copy-cursor',
            'native': 'gateway.input-reclamation.v3.native-cursor'}


def owned_home(db):
    from gateway.runtime_ownership import process_ownership
    from hermes_constants import get_hermes_home
    home = Path(db.db_path).resolve().parent
    if get_hermes_home().resolve() != home or not process_ownership.owns(home):
        raise RuntimeStoreError('permission_denied')
    return home


def copy_path(db, copy):
    if copy['namespace'] not in {'v3', 'native'} or not _DIGEST.fullmatch(copy['digest']):
        raise RuntimeStoreError('storage_unavailable')
    root = _media_root() if copy['namespace'] == 'native' else (
        default_attachment_root(db.db_path) / 'working-documents-v3')
    name = copy['name']
    if not name or Path(name).name != name or name in {'.', '..'}:
        raise RuntimeStoreError('storage_unavailable')
    path = root.resolve() / copy['digest'] / name
    if root.resolve() != root or path.parent.resolve() != path.parent:
        raise RuntimeStoreError('storage_unavailable')
    return path


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


def _external_holds(conn, db, copy, path, now):
    held = _held_media_paths(conn)
    identities = _held_file_identities(held, _media_root())
    if identities is None or str(path) in held:
        return True
    try:
        identity = _file_identity(path)
        if path.stat(follow_symlinks=False).st_nlink != 1:
            return True  # A physical owner outside this copy.
    except FileNotFoundError:
        return False
    if identity in identities:
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


def _collect(db, *, epoch, limit, namespace):
    owned_home(db)
    if type(limit) is not int or not 1 <= limit <= 256:
        raise RuntimeStoreError('invalid_params')
    now = time.time()
    def seal(conn):
        _epoch(conn, epoch)
        cursor_key = _CURSORS[namespace]
        previous = conn.execute('SELECT value FROM state_meta WHERE key=?', (cursor_key,)).fetchone()
        cursor = previous[0] if previous else ''
        rows = conn.execute('''SELECT * FROM input_custody_copies WHERE state!='removed'
            AND namespace=? AND copy_id>? ORDER BY copy_id LIMIT ?''', (namespace, cursor, limit)).fetchall()
        if not rows and cursor:
            rows = conn.execute('''SELECT * FROM input_custody_copies WHERE state!='removed'
                AND namespace=? ORDER BY copy_id LIMIT ?''', (namespace, limit)).fetchall()
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
    """Online owner opportunity: private document copies, never native images."""
    return _collect(db, epoch=epoch, limit=limit, namespace='v3')


def collect_native_inputs(db, *, epoch, limit=64):
    """Parent calls ONLY before the owner accepts input, never from housekeeping.

    Native image targets can be reused by a capture that is not yet admitted.
    Profile ownership/epoch is necessary, not proof of no in-flight captures;
    the startup caller owns that pre-ingress contract.
    """
    return _collect(db, epoch=epoch, limit=limit, namespace='native')
