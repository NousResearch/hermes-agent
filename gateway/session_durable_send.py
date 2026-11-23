"""Scoped, resumable Desktop uploads; admission remains the authority's existing ledger.

No client paths or principal claims. Upload identities include the authenticated subject,
profile, session, submission and occurrence. Bytes survive lost replies and owner restarts.
The bounded staging store is intentionally retained (never age-delete files a queued turn uses).
"""
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from functools import partial

from persistence.core.errors import AuthorityRefusal

CAPABILITY = 'durable-desktop-send-v1'
_ID = re.compile(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}')
_HASH = re.compile(r'[a-f0-9]{64}')
_MAX_FILE = 64 * 1024 * 1024
_MAX_TOTAL = 256 * 1024 * 1024


def handlers(connection):
    return {'durable.describe': partial(describe, connection),
            'durable.create': partial(create, connection),
            'durable.upload': partial(upload, connection),
            'durable.submit': partial(submit, connection)}


async def describe(connection, ref, params):
    if set(params) - {'profile'}:
        raise AuthorityRefusal('invalid_params')
    await connection.ping(ref, {})
    from gateway.session_persistence import authority_persistence
    store = authority_persistence(connection.authority)
    generation = await asyncio.to_thread(store.state_meta.get, 'db_file_generation')
    if not isinstance(generation, str) or not generation:
        raise AuthorityRefusal('storage_unavailable')
    return {'store_id': generation, 'instance_id': connection.authority.instance_id,
            'profile_id': connection.authority.profile_id,
            'principal_id': connection.actor.subject,
            'capabilities': [CAPABILITY, 'durable-admission-v1', 'local-cli-create-v1']}


async def _check_owner(connection, params):
    from gateway.session_persistence import authority_persistence
    generation = await asyncio.to_thread(authority_persistence(connection.authority).state_meta.get, 'db_file_generation')
    expected = {'store_id': generation, 'profile_id': connection.authority.profile_id,
                'principal_id': connection.actor.subject}
    if params.get('owner') != expected or not generation:
        raise AuthorityRefusal('permission_denied')


async def create(connection, ref, params):
    await _check_owner(connection, params)
    if (set(params) - {'profile', 'owner', 'params'} or not isinstance(params.get('params'), dict)
            or 'profile' in params['params']):
        raise AuthorityRefusal('invalid_params')
    from pydantic import ValidationError
    from tui_gateway.contracts.registry import CANONICAL_METHODS
    try:
        CANONICAL_METHODS['session.create'].params.model_validate(params['params'])
    except ValidationError as exc:
        raise AuthorityRefusal('invalid_params') from exc
    result = await connection.create(ref, params['params'])
    return {'session_id': result['session_id']}


def _authorize(connection, ref, params):
    from gateway.session_busy_controls import authorize
    if not ref.session_id or ref.session_id not in connection.subscriptions:
        raise AuthorityRefusal('permission_denied')
    authorize(connection, ref, params, 'session:submit')


def _identity(connection, ref, submission, file):
    if (not isinstance(submission, str) or not _ID.fullmatch(submission)
            or not isinstance(file, dict) or set(file) != {'id', 'name', 'mediaType', 'size', 'sha256'}
            or not isinstance(file['id'], str) or not _ID.fullmatch(file['id'])
            or not isinstance(file['name'], str) or not 0 < len(file['name']) <= 1024
            or '\x00' in file['name'] or not isinstance(file['mediaType'], str)
            or not 0 < len(file['mediaType']) <= 256
            or type(file['size']) is not int or not 0 <= file['size'] <= _MAX_FILE
            or not isinstance(file['sha256'], str) or not _HASH.fullmatch(file['sha256'])):
        raise AuthorityRefusal('invalid_params')
    identity = [connection.authority.profile_id, connection.actor.subject,
                ref.session_id, submission, file['id']]
    key = hashlib.sha256(json.dumps(identity, ensure_ascii=True).encode()).hexdigest()
    return key, json.dumps(file, sort_keys=True, ensure_ascii=True)


def _private_root(home, leaf):
    from hermes_dir_acl import make_user_dir
    root = Path(home).resolve()
    for part in ('cache', leaf):
        root = root / part
        if root.is_symlink():
            raise AuthorityRefusal('storage_unavailable')
        make_user_dir(root, mode=0o700, parents=False, exist_ok=True)
        if not root.is_dir():
            raise AuthorityRefusal('storage_unavailable')
    return root


def _database(home):
    root = _private_root(home, 'durable-desktop-uploads')
    name = root / 'uploads.sqlite'
    if name.is_symlink() or (name.exists() and (not name.is_file() or name.stat().st_nlink != 1)):
        raise AuthorityRefusal('storage_unavailable')
    fd = os.open(name, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    os.close(fd)
    db = sqlite3.connect(name, timeout=30)
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS uploads (id TEXT PRIMARY KEY, metadata TEXT NOT NULL, '
               'size INTEGER NOT NULL, body BLOB NOT NULL)')
    db.commit()
    return db


def _append(home, key, metadata, file, offset, data):
    db = _database(home)
    try:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT metadata, body FROM uploads WHERE id=?', (key,)).fetchone()
        if row is None:
            reserved = db.execute('SELECT COALESCE(SUM(size), 0) FROM uploads').fetchone()[0]
            count = db.execute('SELECT COUNT(*) FROM uploads').fetchone()[0]
            if reserved + file['size'] > _MAX_TOTAL or count >= 4096:
                raise AuthorityRefusal('storage_unavailable')
            body = b''
        else:
            if row[0] != metadata:
                raise AuthorityRefusal('admission_conflict', field='request_id')
            body = bytes(row[1])
        if offset > len(body) or offset + len(data) > file['size']:
            raise AuthorityRefusal('invalid_params')
        overlap = min(len(data), len(body) - offset)
        if body[offset:offset + overlap] != data[:overlap]:
            raise AuthorityRefusal('admission_conflict', field='request_id')
        body += data[overlap:]
        complete = len(body) == file['size']
        if complete and hashlib.sha256(body).hexdigest() != file['sha256']:
            raise AuthorityRefusal('invalid_params')
        db.execute('INSERT OR REPLACE INTO uploads VALUES (?, ?, ?, ?)',
                   (key, metadata, file['size'], body))
        db.commit()
        return {'reference': key, 'sha256': file['sha256'], 'size': file['size'],
                'offset': len(body), 'complete': complete}
    finally:
        db.close()


async def upload(connection, ref, params):
    await _check_owner(connection, params)
    _authorize(connection, ref, params)
    if set(params) - {'session_id', 'profile', 'owner', 'submission_id', 'file', 'offset', 'data'}:
        raise AuthorityRefusal('invalid_params')
    key, metadata = _identity(connection, ref, params.get('submission_id'), params.get('file'))
    offset, encoded = params.get('offset'), params.get('data')
    if type(offset) is not int or offset < 0 or not isinstance(encoded, str) or len(encoded) > 1_398_104:
        raise AuthorityRefusal('invalid_params')
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise AuthorityRefusal('invalid_params') from exc
    return await asyncio.to_thread(_append, connection.authority.profile_id, key, metadata,
                                   params['file'], offset, data)


def _materialize(home, key, metadata, file):
    from gateway.session_ingress_media import sniff_image_mime
    from utils import fsync_directory
    import tempfile
    db = _database(home)
    try:
        row = db.execute('SELECT metadata, body FROM uploads WHERE id=?', (key,)).fetchone()
    finally:
        db.close()
    if row is None or row[0] != metadata:
        raise AuthorityRefusal('not_found')
    body = bytes(row[1])
    if len(body) != file['size'] or hashlib.sha256(body).hexdigest() != file['sha256']:
        raise AuthorityRefusal('invalid_params')
    image = file['mediaType'].startswith('image/')
    if image and sniff_image_mime(body) != file['mediaType']:
        raise AuthorityRefusal('invalid_params')
    root = _private_root(home, 'images' if image else 'durable-desktop-uploads')
    # Preserve a safe extension for document parsers, never a user-controlled path.
    suffix = Path(file['name']).suffix
    suffix = suffix if re.fullmatch(r'\.[a-zA-Z0-9]{1,12}', suffix) else ''
    target = root / (f'durable_{key}_{file["sha256"]}' + suffix)
    fd, temporary = tempfile.mkstemp(prefix='.durable-', dir=root)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        fsync_directory(root)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(target), image


async def submit(connection, ref, params):
    await _check_owner(connection, params)
    _authorize(connection, ref, params)
    if (set(params) - {'session_id', 'profile', 'owner', 'submission_id', 'text', 'attachments'}
            or not isinstance(params.get('text'), str) or len(params['text'].encode()) > 256 * 1024
            or not isinstance(params.get('attachments'), list) or len(params['attachments']) > 32
            or not isinstance(params.get('submission_id'), str) or not _ID.fullmatch(params['submission_id'])):
        raise AuthorityRefusal('invalid_params')
    text, images, seen = params['text'], [], set()
    for attachment in params['attachments']:
        if not isinstance(attachment, dict) or 'reference' not in attachment:
            raise AuthorityRefusal('invalid_params')
        file = {k: v for k, v in attachment.items() if k != 'reference'}
        key, metadata = _identity(connection, ref, params.get('submission_id'), file)
        if key != attachment['reference'] or key in seen:
            raise AuthorityRefusal('permission_denied')
        seen.add(key)
        path, image = await asyncio.to_thread(_materialize, connection.authority.profile_id, key, metadata, file)
        if image:
            images.append({'path': path, 'mime': file['mediaType']})
        else:
            # Same @file wire reference as file.attach, with an immutable owned path.
            text += '\n@file:' + json.dumps(path, ensure_ascii=False)
    if len(images) > 10:
        raise AuthorityRefusal('invalid_params')
    payload = {'session_id': ref.session_id, 'submission_id': params.get('submission_id'), 'text': text}
    if images:
        payload['attachments'] = images
    # Existing authorization, payload conflict detection and UNKNOWN fence are authoritative.
    return await connection.submit(ref, payload)
