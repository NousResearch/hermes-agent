"""Bounded native post-deny reclamation of unambiguous private staging."""
from contextlib import ExitStack
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import secrets
import stat

from gateway import hosted_rooms
from gateway.hosted_room_peer import HostedMemberDispatch, _identifier, canonical_attachment_manifest, attachment_manifest_digest
from gateway.platforms.api_server_room_attachments import _batch_key, RoomAttachmentSpool, _existing_connection

logger = logging.getLogger(__name__)
MAX_SCAN_BATCHES = 1024
MAX_CLEAN_BATCHES = 32
MAX_CLEAN_FILES = 128
MAX_CLEAN_BYTES = 64 * 1024 * 1024
_SCOPE = ('room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
          'member_id', 'target_install_id', 'target_profile')
_ORIGIN = {*_SCOPE, 'version', '_token_sha256', 'grant_id', 'issued_at', 'expires_at',
           'status_expires_at', 'receiver_home', 'receiver_epoch', 'receiver_instance_id'}


class CleanupUncertain(ValueError):
    pass


def _object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise CleanupUncertain('duplicate_origin_fields')
    return result


def _origin(row, operation, claims, exact):
    try:
        encoded = row['staging_origin_json']
        if row['staging_origin_ambiguous'] != 0 or not isinstance(encoded, str) or len(encoded) > 4096:
            return None
        origin = json.loads(encoded, object_pairs_hook=_object)
        if not isinstance(origin, dict) or set(origin) != _ORIGIN or type(origin['version']) is not int or origin['version'] != 1:
            return None
        for key in (*_SCOPE, 'grant_id', 'receiver_instance_id'):
            if key != 'authority_epoch':
                _identifier(origin[key], field=key)
        for key in ('authority_epoch', 'receiver_epoch'):
            if type(origin[key]) is not int or not 1 <= origin[key] <= 2**63 - 1:
                return None
        for key in ('issued_at', 'expires_at', 'status_expires_at'):
            if type(origin[key]) not in (int, float) or not math.isfinite(origin[key]) or origin[key] <= 0:
                return None
        if not origin['issued_at'] < origin['expires_at'] <= origin['status_expires_at']:
            return None
        token = origin['_token_sha256']
        if not isinstance(token, str) or len(token) != 64 or any(c not in '0123456789abcdef' for c in token):
            return None
        if exact and token != claims['_token_sha256']:
            return None
        if origin['receiver_home'] != str(operation.home):
            return None
        if any(origin[key] != row[key] or origin[key] != claims[key] for key in _SCOPE):
            return None
        dispatch = HostedMemberDispatch.from_mapping(json.loads(row['dispatch_json']))
        if _batch_key(dispatch) != row['batch_key'] or any(
                getattr(dispatch, key) != row[key] for key in (*_SCOPE, 'task_id', 'execution_generation')):
            return None
        manifest = canonical_attachment_manifest(json.loads(row['manifest_json']))
        if attachment_manifest_digest(manifest) != row['manifest_digest'] or dispatch.attachment_manifest_digest != row['manifest_digest']:
            return None
        return origin, manifest
    except (ValueError, TypeError, KeyError, OverflowError):
        return None


def _identity(info):
    return info.st_dev, info.st_ino


def _file_identity(info):
    return (*_identity(info), info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_directory(path):
    required = (os.open, os.stat, os.mkdir, os.rename, os.link, os.unlink, os.rmdir)
    if not all(hasattr(os, flag) for flag in ('O_NOFOLLOW', 'O_DIRECTORY', 'O_NONBLOCK')) or not all(fn in os.supports_dir_fd for fn in required):
        raise CleanupUncertain('descriptor_operations_unavailable')
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise CleanupUncertain('noncanonical_root')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class _Quarantine:
    def __init__(self, root):
        self.root = Path(root)
        self.root_fd = _open_directory(root)
        self.name = '.revoke-' + secrets.token_hex(16)
        self.fd = None
        self.files = []
        try:
            os.mkdir(self.name, mode=0o700, dir_fd=self.root_fd)
            self.fd = os.open(self.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.root_fd)
        except BaseException:
            os.close(self.root_fd)
            raise

    def current(self):
        current = _open_directory(self.root)
        try:
            if _identity(os.fstat(current)) != _identity(os.fstat(self.root_fd)):
                raise CleanupUncertain('root_changed')
        finally:
            os.close(current)
        if _identity(os.stat(self.name, dir_fd=self.root_fd, follow_symlinks=False)) != _identity(os.fstat(self.fd)):
            raise CleanupUncertain('quarantine_changed')

    def move(self, name, size, digest, authorize):
        self.current()
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.root_fd)
        recorded = False
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != size:
                raise CleanupUncertain('file_identity_uncertain')
            hasher = hashlib.sha256()
            remaining = size
            while remaining:
                data = os.read(descriptor, min(65536, remaining))
                if not data:
                    raise CleanupUncertain('file_truncated')
                hasher.update(data)
                remaining -= len(data)
            if os.read(descriptor, 1) or hasher.hexdigest() != digest:
                raise CleanupUncertain('file_digest_changed')
            authorize()
            self.current()
            current = os.stat(name, dir_fd=self.root_fd, follow_symlinks=False)
            if (_identity(current), current.st_size, current.st_mtime_ns, current.st_ctime_ns, current.st_nlink) != (
                    _identity(info), info.st_size, info.st_mtime_ns, info.st_ctime_ns, 1):
                raise CleanupUncertain('file_changed')
            destination = str(len(self.files))
            os.rename(name, destination, src_dir_fd=self.root_fd, dst_dir_fd=self.fd)
            self.files.append((name, destination, descriptor, _file_identity(info)))
            recorded = True
            moved = os.stat(destination, dir_fd=self.fd, follow_symlinks=False)
            if (_identity(moved), moved.st_size, moved.st_mtime_ns, moved.st_nlink) != (
                    _identity(info), info.st_size, info.st_mtime_ns, 1):
                raise CleanupUncertain('moved_file_changed')
            self.files[-1] = (name, destination, descriptor, _file_identity(moved))
        finally:
            if not recorded:
                os.close(descriptor)

    def _matches(self, name, identity):
        info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        return stat.S_ISREG(info.st_mode) and _file_identity(info) == identity and info.st_nlink == 1

    def restore(self):
        self.current()
        for source, name, _, identity in reversed(self.files):
            if not self._matches(name, identity):
                raise CleanupUncertain('restore_identity_uncertain')
            try:
                os.link(name, source, src_dir_fd=self.fd, dst_dir_fd=self.root_fd, follow_symlinks=False)
            except FileExistsError:
                continue
            os.unlink(name, dir_fd=self.fd)

    def purge(self, authorize):
        for _, name, _, identity in self.files:
            authorize()
            self.current()
            if not self._matches(name, identity):
                raise CleanupUncertain('purge_identity_uncertain')
            os.unlink(name, dir_fd=self.fd)

    def close(self):
        try:
            self.current()
            os.rmdir(self.name, dir_fd=self.root_fd)
        except (OSError, CleanupUncertain):
            logger.warning('Native staging cleanup retained quarantine %s', self.name)
        finally:
            for _, _, descriptor, _ in self.files:
                os.close(descriptor)
            os.close(self.fd)
            os.close(self.root_fd)


def _denied(shared, owner, origin, exact):
    for conn in (shared, owner):
        if not hosted_rooms.room_grant_is_revoked_on_conn(conn, claims=origin):
            return False
        if not exact:
            row = conn.execute('SELECT revoked_before FROM hosted_room_revoked_grants WHERE scope_key=? AND expires_at>?',
                (hosted_rooms._room_grant_scope_key(origin), hosted_rooms.time.time())).fetchone()
            if row is None or origin['issued_at'] > row['revoked_before']:
                return False
    return True


def _collect(operation, claims, exact, spool, identities, root_identity):
    with ExitStack() as stack:
        shared = spool._connect()
        stack.callback(shared.close)
        shared.execute('BEGIN IMMEDIATE')
        owner = _existing_connection(operation.paths[1])
        stack.callback(owner.close)
        owner.execute('BEGIN IMMEDIATE')

        def authorize():
            operation.require_current(owner)
            root_fd = _open_directory(spool.root)
            try:
                if _identity(os.fstat(root_fd)) != root_identity:
                    raise CleanupUncertain('root_changed')
            finally:
                os.close(root_fd)
            if Path(spool.db_path).resolve() != operation.paths[0] or spool.root != operation.home / 'roomlink-attachment-spool':
                raise CleanupUncertain('spool_changed')
            for conn, path, identity in zip((shared, owner), operation.paths, identities):
                if (Path(conn.execute('PRAGMA database_list').fetchone()[2]).resolve() != path
                        or _identity(path.stat(follow_symlinks=False)) != identity):
                    raise CleanupUncertain('database_changed')

        authorize()
        quarantine = _Quarantine(spool.root)
        stack.callback(quarantine.close)
        count = file_count = byte_count = 0
        origins = []
        try:
            rows = shared.execute('SELECT * FROM roomlink_attachment_batches ORDER BY batch_key LIMIT ?',
                                  (MAX_SCAN_BATCHES,)).fetchall()
            for row in rows:
                evidence = _origin(row, operation, claims, exact)
                if evidence is None:
                    continue
                origin, manifest = evidence
                authorize()
                if not _denied(shared, owner, origin, exact):
                    continue
                entries = shared.execute('SELECT * FROM roomlink_attachment_files WHERE batch_key=? LIMIT ?',
                                         (row['batch_key'], MAX_CLEAN_FILES + 1)).fetchall()
                expected = {item['attachment_id']: (item['size'], item['sha256']) for item in manifest}
                if len(entries) != len(expected) or any(expected.get(e['attachment_id']) != (e['size'], e['sha256']) for e in entries):
                    continue
                # Retained empty attempts are recovery metadata, not new work.
                if not any(e['stored'] for e in entries):
                    continue
                size = sum(e['size'] for e in entries if e['stored'])
                if count >= MAX_CLEAN_BATCHES or file_count + len(entries) > MAX_CLEAN_FILES or byte_count + size > MAX_CLEAN_BYTES:
                    break
                def authorize_denial():
                    authorize()
                    if not shared.in_transaction or not _denied(shared, owner, origin, exact):
                        raise CleanupUncertain('denial_expired')

                for entry in entries:
                    name = spool._file_path(row['batch_key'], entry['attachment_id']).name
                    if entry['stored']:
                        quarantine.move(name, entry['size'], entry['sha256'], authorize_denial)
                    else:
                        try:
                            os.stat(name, dir_fd=quarantine.root_fd, follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        raise CleanupUncertain('uncommitted_file_present')
                authorize()
                if not _denied(shared, owner, origin, exact):
                    raise CleanupUncertain('denial_expired')
                # Bearer denial is not attempt disposal. Keep the immutable
                # manifest and origin so a replacement can resume this generation.
                # Availability resets commit (or roll back) with quarantine moves.
                shared.execute('UPDATE roomlink_attachment_files SET stored=0 WHERE batch_key=?', (row['batch_key'],))
                shared.execute('UPDATE roomlink_attachment_batches SET complete=0 WHERE batch_key=?', (row['batch_key'],))
                origins.append(origin)
                count += 1
                file_count += len(entries)
                byte_count += size
            authorize()
            if not shared.in_transaction or not all(_denied(shared, owner, origin, exact) for origin in origins):
                raise CleanupUncertain('denial_expired')
            shared.commit()
        except BaseException:
            if shared.in_transaction:
                try:
                    quarantine.restore()
                finally:
                    shared.rollback()
            raise
        quarantine.purge(authorize)
        owner.commit()
        return count


def dispatch_native_revoke(connection, method, params):
    from gateway.session_group_peers import _native_owner, _revoke
    from gateway.hosted_room_peer import gateway_room_grant_secret, decode_room_grant
    operation = _native_owner(connection)
    exact = method == 'groups.peer.revoke_exact'
    with ExitStack() as storage:
        identities = None
        try:
            root = operation.home / 'roomlink-attachment-spool'
            descriptor = _open_directory(root)
            storage.callback(os.close, descriptor)
            root_identity = _identity(os.fstat(descriptor))
            held = []
            for path in operation.paths:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                storage.callback(os.close, descriptor)
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode):
                    raise CleanupUncertain('database_identity_uncertain')
                held.append(_identity(info))
            identities = tuple(held)
        except (OSError, CleanupUncertain):
            pass
        result = _revoke(operation, params, exact=exact)
        if identities is None:
            return result
        try:
            operation.require_current()
            claims = decode_room_grant(gateway_room_grant_secret(), params['grant'], permission='status',
                                       allow_expired_for_revocation=True)
            spool = RoomAttachmentSpool(operation.paths[0], _existing_only=True)
            _collect(operation, claims, exact, spool, identities, root_identity)
        except Exception as exc:
            logger.warning('Native staging cleanup skipped or incomplete (%s)', type(exc).__name__)
        return result
