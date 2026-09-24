"""Existing classic custody through real canonical access, without execution."""
import base64
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio


def seed_bound(fixture, sid):
    """Model an already-bound original producer; never run adoption/restoration."""
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_policy import build_policy
    from dataclasses import asdict
    from datetime import datetime
    from gateway.session_authority import LiveSession
    from gateway.session_contract import SessionRef
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_PREFIX
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid,
                           user_id=fixture.connection.actor.subject, chat_type='dm')
    route = 'agent:main:local:' + sid
    entry = SessionEntry(route, sid, datetime.fromtimestamp(1), datetime.fromtimestamp(1),
                         origin=source, platform=Platform.LOCAL)
    policy = build_policy({'source': 'gui', 'cwd': str(fixture.home),
                           'model': 'fixture', 'toolsets': []}, {})
    receipt = dict(profile_id=str(fixture.home), principal_id=source.user_id,
                   session_id=sid, legacy_session_id=sid, request_id='existing-' + sid,
                   route=route, entry=entry.to_dict(), policy=asdict(policy))
    def seed(conn):
        conn.execute("INSERT INTO sessions(id,source,user_id,session_key,chat_id,started_at) VALUES(?,?,?,?,?,?)",
                     (sid, 'gui', source.user_id, route, sid, 1.0))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (POLICY_PREFIX + sid, json.dumps(receipt)))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (LEGACY_PREFIX + sid, json.dumps({k: receipt[k] for k in
                      ('session_id', 'profile_id', 'principal_id', 'legacy_session_id')})))
    fixture.db._execute_write(seed)
    fixture.authority.sessions[sid] = LiveSession(source, route)
    return SessionRef(str(fixture.home), sid)


@pytest_asyncio.fixture
async def exported(tmp_path, monkeypatch):
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch
    import hashlib
    import sqlite3

    home = tmp_path.resolve()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: home)
    # Original #104198 schema/bytes predate canonical binding. No producer runs.
    data = b'already published original bytes\n'
    export_id = 'ce_' + hashlib.sha256(json.dumps([str(home), 'original-producer', 'old-export'],
        separators=(',', ':')).encode()).hexdigest()
    scope = dict(kind='classic', export_id=export_id, execution_generation=1)
    scope_json = json.dumps(scope, sort_keys=True, separators=(',', ':'))
    scope_key = hashlib.sha256(scope_json.encode()).hexdigest()
    blob = 'blob_' + 'a' * 32
    item = dict(artifact_id='rart_' + 'b' * 32, name='report.txt', kind='file', mime='text/plain',
                size=len(data), sha256=hashlib.sha256(data).hexdigest())
    installation = 'install:' + 'c' * 32
    (home / 'install_id').write_text('c' * 32)
    binding = dict(group_id='old-room', thread_id='thread', issued_at=1.0,
                   prompt_sha256=hashlib.sha256(b'old request').hexdigest(),
                   recipients=[dict(installation=installation, profile='default')])
    with sqlite3.connect(home / 'state.db') as conn:
        conn.executescript("""
            CREATE TABLE classic_output_exports (
                export_id TEXT PRIMARY KEY, profile_home TEXT NOT NULL,
                session_key TEXT NOT NULL, request_id TEXT NOT NULL, generation INTEGER NOT NULL,
                binding TEXT NOT NULL, state TEXT NOT NULL, expires REAL NOT NULL,
                text TEXT NOT NULL DEFAULT '', UNIQUE(profile_home,session_key,request_id));
            CREATE TABLE classic_retired_groups (profile_home TEXT, group_id TEXT,
                PRIMARY KEY(profile_home,group_id));
            CREATE TABLE hosted_room_output_artifacts (
                artifact_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, scope_json TEXT NOT NULL,
                name TEXT NOT NULL, kind TEXT NOT NULL, mime TEXT NOT NULL, size INTEGER NOT NULL,
                sha256 TEXT NOT NULL, blob_name TEXT NOT NULL UNIQUE, created_at REAL NOT NULL,
                acknowledged_at REAL, ack_message_event_id TEXT, receipt_expires_at REAL,
                cleanup_required_at REAL, blob_reclaimed_at REAL, UNIQUE(scope_key,sha256,name));
            CREATE TABLE hosted_room_output_generation_fences (
                lineage_key TEXT PRIMARY KEY, lineage_json TEXT NOT NULL, lineage_identity TEXT,
                max_generation INTEGER NOT NULL, retired_generation INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL);
        """)
        conn.execute('INSERT INTO classic_output_exports VALUES(?,?,?,?,?,?,?,?,?)',
            (export_id, str(home), 'original-producer', 'old-export', 1, json.dumps(binding),
             'published', 2.0, 'private original transcript'))
        conn.execute('INSERT INTO hosted_room_output_artifacts VALUES(?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,NULL,NULL)',
            (item['artifact_id'], scope_key, scope_json, item['name'], item['kind'], item['mime'],
             item['size'], item['sha256'], blob, 1.0))
    blob_path = home / 'hosted-room-artifact-outbox' / 'blobs' / blob
    blob_path.parent.mkdir(parents=True)
    blob_path.write_bytes(data)
    db = SessionDB(home / 'state.db')
    runner = SimpleNamespace(adapters={}, _draining=False)
    authority = SessionAuthority(runner, profile_id=str(home), instance_id='read-test', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='read-test'))
    runner.session_authority = authority
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner', 'provider': 'local',
        'profile_id': str(home), 'instance_id': 'read-test', 'capabilities': ['session:read']})
    fixture = SimpleNamespace(home=home, db=db, authority=authority, connection=connection,
        row={'export_id': export_id, 'generation': 1}, item=item, data=data)
    fixture.ref = seed_bound(fixture, 'original-producer')
    fixture.params = dict(session_id=fixture.ref.session_id, installation=installation, group_id='old-room',
                         export_id=export_id, artifact_id=item['artifact_id'], generation=1)
    try:
        yield fixture
    finally:
        await connection.close()
        db.close()


async def read(fixture, **changes):
    return await fixture.connection.dispatch({'id': 1, 'method': 'session.export.read',
                                              'params': {**fixture.params, **changes}})


def dump(fixture):
    with fixture.db._read_ctx() as conn:
        return tuple(conn.iterdump())


@pytest.mark.asyncio
async def test_published_read_is_non_creating_and_exact(exported, monkeypatch):
    from gateway.hosted_room_artifacts import RoomArtifactOutbox
    before = dump(exported)
    def forbidden(*args, **kwargs):
        pytest.fail('a read must not construct or initialize custody')
    monkeypatch.setattr(RoomArtifactOutbox, '__init__', forbidden)
    monkeypatch.setattr(RoomArtifactOutbox, '_initialize', forbidden)
    result = await read(exported)
    assert 'result' in result, result
    value = result['result']
    assert value['session_id'] == exported.ref.session_id
    assert value['item'] == exported.item
    assert value['export_id'] == exported.row['export_id']
    assert value['generation'] == exported.row['generation']
    assert base64.b64decode(value['content_base64'], validate=True) == exported.data
    assert 'text' not in value and 'path' not in json.dumps(value)
    assert dump(exported) == before
    assert not exported.db._read_all('SELECT * FROM session_admissions')


@pytest.mark.asyncio
@pytest.mark.parametrize('change', [
    {'installation': 'install:other'}, {'group_id': 'another-room'}, {'generation': 2},
    {'generation': True}, {'artifact_id': 'rart_' + '0' * 32}, {'session_id': 'not-bound'},
    {'profile': 'named'}, {'path': '/not/a/read/argument'}, {'request_id': 'old-export'},
])
async def test_exact_selectors_do_not_fall_back(exported, change):
    before = dump(exported)
    assert 'error' in await read(exported, **change)
    assert dump(exported) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['foreign', 'missing_read', 'closed', 'cold', 'named'])
async def test_existing_read_authority_is_required(exported, kind):
    connection, authority = exported.connection, exported.authority
    if kind == 'foreign':
        connection.actor = replace(connection.actor, subject='other')
    elif kind == 'missing_read':
        connection.actor = replace(connection.actor, capabilities=frozenset())
    elif kind == 'closed':
        await connection.close()
    elif kind == 'cold':
        authority.sessions.clear()
    else:
        authority.profile_id = str(exported.home / 'profiles' / 'named')
        connection.actor = replace(connection.actor, profile_id=authority.profile_id)
    before = dump(exported)
    assert 'error' in await read(exported)
    assert dump(exported) == before


@pytest.mark.asyncio
async def test_an_unrelated_authorized_session_cannot_read_the_export(exported):
    other = seed_bound(exported, 'unrelated-producer')
    assert 'error' in await read(exported, session_id=other.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['retire', 'epoch', 'binding', 'delete_session', 'close_transport', 'read_scope'])
async def test_fresh_checks_after_byte_read_refuse_changes(exported, monkeypatch, change):
    from gateway import session_classic_exports as reader
    original = reader._read_bytes
    calls = []
    def changed(*args):
        calls.append(True)
        data = original(*args)
        if change == 'retire':
            exported.db._execute_write(lambda conn: conn.execute(
                "UPDATE classic_output_exports SET state='retired' WHERE export_id=?", (exported.row['export_id'],)))
        elif change == 'epoch':
            from hermes_state_runtime import begin_runtime_epoch
            begin_runtime_epoch(exported.db, instance_id='different-owner')
        elif change == 'binding':
            exported.db._execute_write(lambda conn: conn.execute(
                "UPDATE classic_output_exports SET session_key='unrelated' WHERE export_id=?", (exported.row['export_id'],)))
        elif change == 'delete_session':
            exported.db._execute_write(lambda conn: conn.execute(
                'DELETE FROM sessions WHERE id=?', (exported.ref.session_id,)))
        elif change == 'close_transport':
            exported.authority.events.pop(exported.connection.actor.transport_id)
        else:
            exported.connection.actor = replace(exported.connection.actor, capabilities=frozenset())
        return data
    monkeypatch.setattr(reader, '_read_bytes', changed)
    assert 'error' in await read(exported)
    assert calls == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['missing_schema', 'missing_bytes', 'changed_bytes', 'symlink', 'retirement'])
async def test_unavailable_custody_is_not_repaired(exported, kind):
    row = exported.db._read_one('SELECT blob_name FROM hosted_room_output_artifacts')
    path = exported.home / 'hosted-room-artifact-outbox' / 'blobs' / row['blob_name']
    if kind == 'missing_schema':
        exported.db._execute_write(lambda conn: conn.execute('DROP TABLE classic_retired_groups'))
    elif kind == 'retirement':
        exported.db._execute_write(lambda conn: conn.execute(
            'INSERT INTO classic_retired_groups VALUES (?,?)', (str(exported.home), 'old-room')))
    elif kind == 'symlink':
        other = exported.home / 'outside.txt'
        other.write_bytes(exported.data)
        path.unlink()
        path.symlink_to(other)
    elif kind == 'changed_bytes':
        path.write_bytes(b'x' * len(exported.data))
    else:
        path.unlink()
    before = dump(exported)
    assert 'error' in await read(exported)
    assert dump(exported) == before


@pytest.mark.asyncio
async def test_missing_custody_database_is_not_created(exported):
    # The authority remains bound to the existing connection; no missing-file fallback.
    original = exported.authority.db.db_path
    absent = exported.home / 'absent.db'
    exported.authority.db.db_path = str(absent)
    try:
        assert 'error' in await read(exported)
        assert not absent.exists()
    finally:
        exported.authority.db.db_path = original


@pytest.mark.asyncio
async def test_exact_compression_lineage_is_supported_but_ambiguity_is_not(exported):
    db, sid = exported.db, exported.ref.session_id
    from hermes_state_local import POLICY_PREFIX
    def compress(conn):
        receipt = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?',
                                         (POLICY_PREFIX + sid,)).fetchone()[0])
        conn.execute("UPDATE sessions SET end_reason='compression' WHERE id=?", (sid,))
        conn.execute("INSERT INTO sessions(id,source,user_id,chat_id,parent_session_id,started_at) VALUES(?,?,?,?,?,?)",
                     ('exact-child', 'gui', receipt['principal_id'], sid, sid, 2.0))
        receipt['entry']['session_id'] = 'exact-child'
        receipt['lineage'] = [sid, 'exact-child']
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(receipt), POLICY_PREFIX + sid))
    db._execute_write(compress)
    before = dump(exported)
    assert 'result' in await read(exported)
    assert dump(exported) == before
    db.create_session('competing-child', source='gui', parent_session_id=sid)
    assert 'error' in await read(exported)


@pytest.mark.asyncio
async def test_deleted_or_unbound_original_session_never_gets_adopted(exported, monkeypatch):
    from hermes_state_local import POLICY_PREFIX
    def forbidden(*args, **kwargs):
        pytest.fail('read cannot adopt or restore a session')
    monkeypatch.setattr('gateway.session_local_migration.adopt_legacy_session', forbidden)
    monkeypatch.setattr('gateway.session_local_recovery.restore_local_session', forbidden)
    exported.db._execute_write(lambda conn: conn.execute('DELETE FROM state_meta WHERE key=?',
        (POLICY_PREFIX + exported.ref.session_id,)))
    before = dump(exported)
    assert 'error' in await read(exported)
    assert dump(exported) == before


def test_readonly_opener_does_not_create_missing_storage(tmp_path):
    from gateway.session_classic_exports import _readonly
    missing = tmp_path / 'missing' / 'state.db'
    import sqlite3
    with pytest.raises(sqlite3.OperationalError):
        with _readonly(missing):
            pytest.fail('missing store was created')
    assert not missing.parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [[], ['bad'], 'bad', 1])
async def test_non_object_rpc_is_refused(exported, value):
    reply = await exported.connection.dispatch({'id': 1, 'method': 'session.export.read', 'params': value})
    assert reply['error']['message'] == 'invalid_params'

@pytest.mark.asyncio
@pytest.mark.parametrize('metadata', ['valid', 'non-object'])
async def test_reset_is_not_original_compression_lineage(exported, metadata):
    from hermes_state_local import POLICY_PREFIX
    sid = exported.ref.session_id
    def seed(conn):
        receipt = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?',
                                         (POLICY_PREFIX + sid,)).fetchone()[0])
        conn.execute("UPDATE sessions SET end_reason='session_reset' WHERE id=?", (sid,))
        conn.execute("INSERT INTO sessions(id,source,user_id,chat_id,parent_session_id,model_config,started_at) VALUES(?,?,?,?,?,?,?)",
                     ('reset-child', 'gui', receipt['principal_id'], sid, sid,
                      json.dumps({'_reset_from': sid} if metadata == 'valid' else []), 2.0))
        receipt['entry']['session_id'] = 'reset-child'
        receipt['lineage'] = [sid, 'reset-child']
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(receipt), POLICY_PREFIX + sid))
    exported.db._execute_write(seed)
    before = dump(exported)
    assert (await read(exported))['error']['message'] == 'classic_export_unavailable'
    assert dump(exported) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['retire', 'permission', 'install', 'blob', 'binding', 'fence'])
async def test_fresh_checks_after_async_handoff(exported, monkeypatch, change):
    from gateway import session_classic_exports as reader
    original = reader.asyncio.to_thread
    async def completed(*args, **kwargs):
        loaded = await original(*args, **kwargs)
        if change == 'permission':
            exported.connection.actor = replace(exported.connection.actor, capabilities=frozenset())
        elif change == 'install':
            (exported.home / 'install_id').write_text('d' * 32)
        elif change == 'blob':
            loaded[1].write_bytes(b'x' * len(exported.data))
        elif change == 'binding':
            exported.authority.sessions.clear()
        elif change == 'fence':
            from gateway.hosted_room_artifacts_classic import ClassicExportScope
            scope = ClassicExportScope(exported.row['export_id'], 1)
            exported.db._execute_write(lambda conn: conn.execute(
                'INSERT INTO hosted_room_output_generation_fences VALUES(?,?,?,?,?,?)',
                (scope.lineage_key, scope.lineage_json, scope.lineage_json, 1, 1, 3.0)))
        else:
            exported.db._execute_write(lambda conn: conn.execute(
                "UPDATE classic_output_exports SET state='retired'"))
        return loaded
    monkeypatch.setattr(reader.asyncio, 'to_thread', completed)
    assert 'error' in await read(exported)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['absent', 'symlink', 'malformed'])
async def test_install_identity_is_never_created_or_followed(exported, kind):
    path = exported.home / 'install_id'
    path.unlink()
    if kind == 'symlink':
        other = exported.home / 'foreign-install'
        other.write_text('c' * 32)
        path.symlink_to(other)
    elif kind == 'malformed':
        path.write_text('not-an-install')
    before = dump(exported)
    assert 'error' in await read(exported)
    assert dump(exported) == before
    if kind == 'absent':
        assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('target', ['receipt', 'binding'])
async def test_malformed_stored_identity_is_unavailable(exported, target):
    from hermes_state_local import POLICY_PREFIX
    if target == 'receipt':
        exported.db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?',
            ('[]', POLICY_PREFIX + exported.ref.session_id)))
    else:
        exported.db._execute_write(lambda conn: conn.execute('UPDATE classic_output_exports SET binding=?', ('[]',)))
    assert 'error' in await read(exported)


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['owner-check', 'worker-check'])
async def test_evicted_binding_cannot_enter_implicit_restore(exported, monkeypatch, phase):
    import threading
    from unittest.mock import Mock
    before = dump(exported)
    original = exported.db._raise_if_db_replaced
    evicted = []
    owner_thread = threading.get_ident()
    authorizations = []
    authorize = exported.authority.authorize

    def authorize_on_owner(*args):
        authorizations.append(threading.get_ident())
        return authorize(*args)

    def check_then_evict():
        original()
        if not evicted and (phase == 'owner-check' or threading.get_ident() != owner_thread):
            evicted.append(exported.authority.sessions.pop(exported.ref.session_id))

    restore = Mock(side_effect=AssertionError('a file read entered implicit restoration'))
    monkeypatch.setattr('gateway.session_local_recovery.restore_local_session', restore)
    monkeypatch.setattr(exported.authority, 'authorize', authorize_on_owner)
    monkeypatch.setattr(exported.db, '_raise_if_db_replaced', check_then_evict)
    assert 'error' in await read(exported)
    restore.assert_not_called()
    assert all(thread == owner_thread for thread in authorizations)
    assert evicted and exported.ref.session_id not in exported.authority.sessions
    assert dump(exported) == before


@pytest.mark.asyncio
async def test_missing_artifact_column_is_structured_unavailable(exported, monkeypatch):
    from gateway import session_classic_exports as reader
    exported.db._execute_write(lambda conn: conn.execute(
        'ALTER TABLE hosted_room_output_artifacts DROP COLUMN blob_reclaimed_at'))
    before = dump(exported)

    def forbidden(*args, **kwargs):
        pytest.fail('incompatible custody must refuse before blob reads')

    monkeypatch.setattr(reader, '_read_bytes', forbidden)
    assert (await read(exported))['error']['message'] == 'classic_export_unavailable'
    assert dump(exported) == before
