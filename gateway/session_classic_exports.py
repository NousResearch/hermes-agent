"""Read already-published #104198 custody through existing canonical authority.

No export/session creation, migration, pruning, forwarding or retirement on reads.
"""
import asyncio
import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat

from gateway.config import Platform
from gateway.hosted_room_artifacts import (
    MAX_ATTACHMENT_BYTES, RoomArtifactOutbox, open_room_artifact_path,
)
from gateway.hosted_room_artifacts_classic import ClassicExportScope, identifier
from hermes_state_local import POLICY_PREFIX
from hermes_state_local_lineage import validate_local_lineage
from hermes_state_runtime import RuntimeStoreError, _epoch

_FIELDS = {'session_id', 'installation', 'group_id', 'export_id', 'artifact_id', 'generation'}
_HEX = re.compile(r'[0-9a-f]{64}')
_MIME = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9!#$&^_.+-]*/[a-zA-Z0-9][a-zA-Z0-9!#$&^_.+-]*')


def _unavailable():
    return RuntimeStoreError('classic_export_unavailable')


@contextmanager
def _readonly(path):
    conn = sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True, timeout=1)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA query_only=ON')
        conn.execute('BEGIN')
        yield conn
    finally:
        conn.close()


def _current(connection, ref, epoch):
    from gateway.session_authorities import authority_for_profile_id, served_profile_name
    authority, actor = connection.authority, connection.actor
    home, path = Path(authority.profile_id), Path(authority.db.db_path)
    if (not home.is_absolute() or home != home.resolve() or home.parent.name == 'profiles'
            or served_profile_name(home) != 'default' or path != home / 'state.db'
            or path != path.resolve()):
        raise _unavailable()
    if (authority.epoch != epoch or authority_for_profile_id(authority.runner, authority.profile_id) is not authority
            or authority.events.get(actor.transport_id) is not connection.transport
            or ref.session_id not in authority.sessions):
        raise _unavailable()
    authority.db._raise_if_db_replaced()
    live = authority.sessions.get(ref.session_id)
    if live is None or live.source is None or live.source.platform != Platform.LOCAL:
        raise _unavailable()
    # Same owner's existing identity only: never mint/cache or follow a link.
    with open_room_artifact_path(home / 'install_id') as (_, fd):
        before = os.fstat(fd)
        identity = os.read(fd, 34).decode('ascii').strip()
        if (not re.fullmatch(r'[0-9a-f]{32}', identity) or before.st_size > 33
                or _stamp(os.fstat(fd)) != _stamp(before)):
            raise _unavailable()
    return home, path, 'install:' + identity, live


def _authorize_bound(connection, ref, epoch):
    # Called only on the owner-loop side, with no await between the existing
    # binding gate and authorization. Worker snapshots never call the general
    # authorizer: its missing-session path is allowed to restore live state.
    _current(connection, ref, epoch)
    authority = connection.authority
    if ref.session_id not in authority.sessions:
        raise _unavailable()
    authority.authorize(connection.actor, ref, 'session:read')


def _lineage(conn, authority, ref, live):
    raw = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + ref.session_id,)).fetchone()
    if raw is None or len(raw[0]) > 1024 * 1024:
        raise _unavailable()
    receipt = json.loads(raw[0])
    if not isinstance(receipt, dict):
        raise _unavailable()
    chain = receipt.get('lineage', [receipt.get('legacy_session_id', ref.session_id)])
    if (receipt['profile_id'] != authority.profile_id or receipt['session_id'] != ref.session_id
            or receipt['principal_id'] != live.source.user_id or receipt['route'] != live.route
            or receipt['policy']['source'] not in {'cli', 'tui', 'gui'}
            or not isinstance(chain, list) or not 1 <= len(chain) <= 101
            or len(set(chain)) != len(chain)):
        raise _unavailable()
    identities = []
    for index, sid in enumerate(chain):
        row = conn.execute('SELECT id,chat_id,user_id,parent_session_id,end_reason,source,started_at '
                           'FROM sessions WHERE id=?', (sid,)).fetchone()
        if (row is None or row['chat_id'] != ref.session_id or row['user_id'] != receipt['principal_id']
                or row['source'] not in {'cli', 'tui', 'gui'}):
            raise _unavailable()
        # Current canonical history also supports reset edges, but this read
        # subset permits only the original producer's compression lineage.
        if index + 1 < len(chain) and row['end_reason'] != 'compression':
            raise _unavailable()
        if row['end_reason'] == 'compression':
            # The usual history picker may prefer one child; a custody read must
            # instead refuse competing continuation candidates.
            children = conn.execute("SELECT id FROM sessions WHERE parent_session_id=? "
                "AND json_extract(model_config,'$._branched_from') IS NULL "
                "AND json_extract(model_config,'$._delegate_from') IS NULL "
                "AND COALESCE(source,'')!='tool' LIMIT 2", (sid,)).fetchall()
            if index + 1 == len(chain) or len(children) != 1 or children[0][0] != chain[index + 1]:
                raise _unavailable()
        identities.append(tuple(row))
    # Reject reset/non-compression edges before the shared validator examines
    # reset metadata. This reader intentionally supports a narrower lineage.
    validate_local_lineage(conn, receipt)
    return chain, (raw[0], identities)


def _snapshot(connection, ref, params, epoch):
    home, path, installation, live = _current(connection, ref, epoch)
    if params['installation'] != installation:
        raise _unavailable()
    with _readonly(path) as conn:
        _epoch(conn, epoch)
        if conn.execute('SELECT instance_id FROM runtime_epoch WHERE singleton=1').fetchone()[0] != connection.authority.instance_id:
            raise _unavailable()
        chain, lineage = _lineage(conn, connection.authority, ref, live)
        row = conn.execute('SELECT export_id,profile_home,session_key,request_id,generation,binding,state,expires '
                           'FROM classic_output_exports WHERE export_id=?', (params['export_id'],)).fetchone()
        if (row is None or row['profile_home'] != str(home) or row['session_key'] not in chain
                or row['generation'] != params['generation'] or row['state'] != 'published'
                or len(row['binding']) > 32768):
            raise _unavailable()
        binding = json.loads(row['binding'])
        recipients = binding['recipients']
        if (binding['group_id'] != params['group_id'] or not isinstance(recipients, list)
                or not 1 <= len(recipients) <= 6 or any(not isinstance(r, dict)
                or set(r) != {'installation', 'profile'} for r in recipients)):
            raise _unavailable()
        for recipient in recipients:
            for value in recipient.values():
                identifier(value)
        if conn.execute('SELECT 1 FROM classic_retired_groups WHERE profile_home=? AND group_id=?',
                        (str(home), params['group_id'])).fetchone():
            raise _unavailable()
        scope = ClassicExportScope(row['export_id'], row['generation'])
        retired = conn.execute('SELECT retired_generation FROM hosted_room_output_generation_fences '
                              'WHERE lineage_identity=?', (scope.lineage_json,)).fetchone()
        if retired is not None and retired[0] >= scope.execution_generation:
            raise _unavailable()
        artifact = conn.execute('SELECT artifact_id,scope_key,scope_json,name,kind,mime,size,sha256,blob_name,'
                               'created_at,acknowledged_at,ack_message_event_id,receipt_expires_at,'
                               'cleanup_required_at,blob_reclaimed_at FROM hosted_room_output_artifacts '
                               'WHERE artifact_id=? AND scope_key=?',
                               (params['artifact_id'], scope.key)).fetchone()
        if (artifact is None or json.loads(artifact['scope_json']) != scope.as_mapping()
                or any(artifact[k] is not None for k in ('acknowledged_at', 'cleanup_required_at', 'blob_reclaimed_at'))
                or type(artifact['size']) is not int or not 0 < artifact['size'] <= MAX_ATTACHMENT_BYTES
                or not _HEX.fullmatch(artifact['sha256'])
                or not re.fullmatch(r'blob_[0-9a-f]{32}', artifact['blob_name'])
                or artifact['kind'] not in {'file', 'image', 'pdf'}
                or len(artifact['mime']) > 127 or not _MIME.fullmatch(artifact['mime'])
                or (artifact['kind'] == 'image' and not artifact['mime'].startswith('image/'))
                or (artifact['kind'] == 'pdf' and artifact['mime'] != 'application/pdf')
                or artifact['name'] != RoomArtifactOutbox._safe_name(artifact['name'])):
            raise _unavailable()
        item = RoomArtifactOutbox._manifest(artifact)
        result = dict(session_id=ref.session_id, installation=installation, export_id=row['export_id'],
            group_id=binding['group_id'], generation=row['generation'], state='published',
            recipients=recipients, item=item)
        proof = (lineage, tuple(row), tuple(artifact))
        return result, home / 'hosted-room-artifact-outbox' / 'blobs' / artifact['blob_name'], proof


def _stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_bytes(path, size, digest):
    with open_room_artifact_path(path) as (_, fd):
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise _unavailable()
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(MAX_ATTACHMENT_BYTES + 1)
        if _stamp(os.fstat(fd)) != _stamp(before) or len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise _unavailable()
        return data, _stamp(before)


def _load(connection, ref, params, epoch):
    result, path, proof = _snapshot(connection, ref, params, epoch)
    data, stamp = _read_bytes(path, result['item']['size'], result['item']['sha256'])
    current, current_path, current_proof = _snapshot(connection, ref, params, epoch)
    if current != result or current_path != path or current_proof != proof:
        raise _unavailable()
    result['content_base64'] = base64.b64encode(data).decode('ascii')
    return result, path, stamp, proof


async def read_classic_export(connection, ref, params):
    if not isinstance(params, dict) or not _FIELDS <= set(params) <= _FIELDS | {'profile'}:
        raise RuntimeStoreError('invalid_params')
    try:
        for key in _FIELDS - {'generation'}:
            identifier(params[key])
        if (type(params['generation']) is not int or params['generation'] < 1
                or not re.fullmatch(r'ce_[0-9a-f]{64}', params['export_id'])
                or not re.fullmatch(r'rart_[0-9a-f]{32}', params['artifact_id'])
                or ('profile' in params and params['profile'] != 'default')):
            raise ValueError('invalid selectors')
    except (ValueError, TypeError) as exc:
        raise RuntimeStoreError('invalid_params') from exc
    params = dict(params)
    epoch = connection.authority.epoch
    try:
        _authorize_bound(connection, ref, epoch)
        result, path, stamp, proof = await asyncio.to_thread(_load, connection, ref, params, epoch)
        _authorize_bound(connection, ref, epoch)
        with open_room_artifact_path(path) as (_, fd):
            if _stamp(os.fstat(fd)) != stamp:
                raise _unavailable()
        # Reopen AFTER byte I/O and the async handoff. The first pinned snapshot
        # is never reused as evidence of current authority or retirement state.
        current, current_path, current_proof = _snapshot(connection, ref, params, epoch)
        if current_path != path or current_proof != proof or current != {k: v for k, v in result.items() if k != 'content_base64'}:
            raise _unavailable()
        return result
    except RuntimeStoreError:
        raise
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as exc:
        raise _unavailable() from exc
