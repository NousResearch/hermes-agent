"""Owned root Files target, native consent and inert HTTP admission (no runtime)."""
import pytest

from tests.gateway.test_canonical_peer_target_setup import target, invite  # noqa: F401


@pytest.fixture
def files_target(target):
    from gateway.runtime_ownership import process_ownership
    from gateway.hosted_room_input_custody import initialize_input_custody
    from gateway.hosted_room_input_reclamation import initialize_working_copies
    process_ownership.reserve([target.home])
    try:
        initialize_input_custody(target.db)
        initialize_working_copies(target.db, epoch=target.authority.epoch)
        yield target
    finally:
        process_ownership.release(target.home)


@pytest.mark.asyncio
async def test_native_files_rights_require_real_owned_initialized_custody(files_target):
    from gateway.hosted_room_peer import decode_room_grant
    invitation = await invite(files_target)
    assert invitation['catalog']['attachments'] is True
    claims = decode_room_grant(files_target.adapter._room_grant_secret(), invitation['grant'],
                               permission='attachment.stage')
    assert set(claims['permissions']) == {'approve', 'dispatch', 'status', 'stop', 'attachment.stage'}
    replay = await invite(files_target)
    assert replay['grant'] == invitation['grant']


def source_files(home, *, mixed=False):
    from gateway import hosted_rooms
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from gateway.session_hosted_attachments import append_user_event
    from types import SimpleNamespace
    import base64
    db = home / 'source.db'
    hosted_rooms.create_room(db, room_id='room-one', name='Source Files',
        members=[{'member_id': 'member-one', 'profile': 'default', 'handle': 'one', 'display_name': 'One'}],
        authority_gateway_id='home-gateway')
    store = HostedRoomAttachmentStore(db)
    raw = [('first.txt', 'file', 'text/plain', b'first independent document'),
           ('second.txt', 'file', 'text/plain', b'second independent document')]
    if mixed:
        raw[1] = ('pixel.png', 'image', 'image/png', base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII='))
    uploads = [store.put(room_id='room-one', upload_id=f'upload-{n}', name=name, kind=kind, mime=mime, data=data)
               for n, (name, kind, mime, data) in enumerate(raw)]
    manifest = [{key: item[key] for key in ('attachment_id', 'kind', 'name', 'size', 'mime')} for item in uploads]
    service = SimpleNamespace(db_path=db, _room=lambda room: hosted_rooms.room_state(db, room_id=room))
    event = append_user_event(service, room_id='room-one', event_id='event-one',
        payload={'text': 'original prompt', 'thread_id': 'thread-one', 'attachments': manifest},
        gateway_id='home-gateway', epoch=1)
    return store, event, [dict(item, event_id='event-one') for item in manifest], [item[3] for item in raw]


def inprocess_http(target, monkeypatch):
    """Replace only the socket boundary, exercising actual client's HTTP bytes."""
    import asyncio
    import io
    import json
    import urllib.error
    from urllib.parse import urlsplit, unquote
    from aiohttp.test_utils import make_mocked_request
    from unittest.mock import AsyncMock
    from tui_gateway import hosted_room_peer_http
    loop = asyncio.get_running_loop()
    calls = []
    table = target.adapter._http_route_table()
    async def send(wire):
        path, method = urlsplit(wire.full_url).path, wire.get_method()
        calls.append((method, path))
        match = {}
        handler = None
        for verb, template, candidate in table:
            if verb != method:
                continue
            chunks, wanted = path.split('/'), template.split('/')
            if len(chunks) != len(wanted):
                continue
            if all(a == z or z.startswith('{') for a, z in zip(chunks, wanted)):
                match = {z[1:-1]: unquote(a) for a, z in zip(chunks, wanted) if z.startswith('{')}
                handler = candidate
                break
        assert handler is not None, (method, path)
        data = wire.data or b''
        if not isinstance(data, bytes):
            data = b''.join(data)
        class Content:
            async def iter_chunked(self, size):
                for offset in range(0, len(data), size):
                    yield data[offset:offset + size]
        req = make_mocked_request(method, path, headers=dict(wire.header_items()), match_info=match,
                                  payload=Content())
        req.json = AsyncMock(return_value=json.loads(data) if method == 'POST' and data else {})
        response = await handler(req)
        if response.status >= 400:
            raise urllib.error.HTTPError(wire.full_url, response.status, response.reason,
                                         response.headers, io.BytesIO(response.body))
        return io.BytesIO(response.body)
    def open_wire(wire, **kwargs):
        return asyncio.run_coroutine_threadsafe(send(wire), loop).result(10)
    monkeypatch.setattr(hosted_room_peer_http, '_open_roomlink_url', open_wire)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize('mixed', [False, True])
async def test_real_source_client_batch_admission_replay_and_disposal(files_target, monkeypatch, mixed):
    import asyncio
    import json
    from pathlib import Path
    from tui_gateway.hosted_room_driver import HostedRoomBinding, ROOM_SESSION_SOURCE
    from tui_gateway.hosted_room_peer_transport import PeerMemberRoute, PeerHostedRoomTransport
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.platforms import api_server_runs
    from gateway.session_api_turn import prepare_api_execution
    target = files_target
    issued = await invite(target)
    store, event, attachments, raw = source_files(target.home, mixed=mixed)
    catalog = issued['catalog']
    route = PeerMemberRoute(home_install_id='home-install', member_id='member-one',
        target_install_id=catalog['installation_id'], target_profile='default',
        capability_digest=catalog['catalog_digest'], cancellation_scope_id='cancel-one', trace_id='trace-one',
        grant=issued['grant'], execution_policy_digest=catalog['execution_policy']['policy_digest'],
        attachments=catalog['attachments'])
    client = PeerRunsHTTPClient(base_url='http://127.0.0.1:8642', api_key='')
    transport = PeerHostedRoomTransport(binding=HostedRoomBinding('room-one', 'home-gateway', 1),
        route=route, client=client, source_event_seq=event['seq'], task_id='task-one',
        execution_generation=1, attachment_store=store)
    calls = inprocess_http(target, monkeypatch)
    launched = []
    async def inert(adapter, launch, **kwargs):
        launched.append(launch.admission)
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    result = await asyncio.to_thread(transport.submit, profile='default', session_id='peer-session',
        prompt='original prompt', source=ROOM_SESSION_SOURCE, task=TaskIdentity('room-one', 'task-one', 'thread-one', 'turn-one'),
        execution_generation=1, on_terminal=lambda _: None, attachments=attachments)
    await asyncio.sleep(0)
    assert result
    assert calls.count(('POST', '/v1/room-members/attachments')) == 1
    assert len([c for c in calls if c[0] == 'PUT']) == 2
    assert len(launched) == 1
    authority, ref, row = launched[0]
    assert authority is target.authority and row['payload']['text'] == 'original prompt'
    refs = row['payload']['api_turn_v1']['settings']['room_input_media']['media']
    assert [Path(r['path']).read_bytes() for r in refs] == raw
    assert sum('working-documents-v3' in r['path'] for r in refs) == (1 if mixed else 2)
    assert target.db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == (1 if mixed else 2)
    assert issued['grant'] not in json.dumps(row)
    prepared = prepare_api_execution(authority, ref, row['payload'])
    assert ('shared files' in prepared['content']) if not mixed else isinstance(prepared['content'], list)
    saved = dict(target.db._conn.execute('SELECT * FROM session_admissions').fetchone())
    await asyncio.to_thread(client.discard_attachments, task_id='task-one', execution_generation=1, grant=route.grant)
    assert not target.db._conn.execute('SELECT 1 FROM input_custody_refs WHERE generation<1').fetchone()
    from gateway.platforms.api_server_room_attachments import _default_spool
    spool = _default_spool()
    with spool._transaction() as conn:
        assert conn.execute('SELECT count(*) FROM roomlink_attachment_batches').fetchone()[0] == 0
    def forbidden(*args, **kwargs):
        pytest.fail('accepted replay touched the discarded spool')
    monkeypatch.setattr(spool, 'materialize', forbidden)
    replay = await asyncio.to_thread(client.dispatch, dispatch=transport._dispatch.as_mapping(), grant=route.grant)
    assert replay
    assert len(launched) == 1
    from dataclasses import replace
    from tests.gateway.test_canonical_peer_text_admission import run_request
    conflict = await target.adapter._handle_runs(run_request(route.grant,
        replace(transport._dispatch, attachment_manifest_digest='0' * 64)))
    assert conflict.status == 409, conflict.text
    assert dict(target.db._conn.execute('SELECT * FROM session_admissions').fetchone()) == saved
    assert [Path(r['path']).read_bytes() for r in refs] == raw
    assert prepare_api_execution(authority, ref, row['payload'])['content'] == prepared['content']
    from tests.gateway.test_canonical_peer_text_admission import dispatch, run_request
    text = await target.adapter._handle_runs(run_request(route.grant, dispatch(issued, task='next-text', prompt='clean text')))
    assert text.status == 202, text.text
    await asyncio.sleep(0)
    later = launched[1]
    assert prepare_api_execution(*later[:2], later[2]['payload'])['content'] == 'clean text'
    assert 'room_input_media' not in later[2]['payload']['api_turn_v1']['settings']


@pytest.mark.asyncio
@pytest.mark.parametrize('denial', ['ownership', 'uninitialized', 'named', 'detached'])
async def test_unavailable_files_never_initialize_or_bind(target, monkeypatch, denial):
    from gateway.platforms.api_server_room_grants import _local_room_catalog
    from gateway import hosted_rooms
    from gateway.runtime_ownership import process_ownership
    before = list(target.db._conn.execute("SELECT name FROM sqlite_master ORDER BY name"))
    if denial == 'ownership':
        assert not process_ownership.owns(target.home)
    if denial == 'detached':
        target.runner.adapters.clear()
    _, catalog = _local_room_catalog(target.adapter, 'other' if denial == 'named' else 'default',
                                     hosted_rooms.local_authority_gateway_id())
    assert catalog['attachments'] is False
    assert list(target.db._conn.execute("SELECT name FROM sqlite_master ORDER BY name")) == before
    assert target.db._conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_native_old_four_rights_cannot_stage_after_initialization(target):
    from gateway.runtime_ownership import process_ownership
    from gateway.hosted_room_input_custody import initialize_input_custody
    from gateway.hosted_room_input_reclamation import initialize_working_copies
    from gateway.hosted_room_peer import decode_room_grant, HostedRoomGrantError
    issued = await invite(target)
    assert issued['catalog']['attachments'] is False
    process_ownership.reserve([target.home])
    try:
        initialize_input_custody(target.db)
        initialize_working_copies(target.db, epoch=target.authority.epoch)
        with pytest.raises(HostedRoomGrantError):
            decode_room_grant(target.adapter._room_grant_secret(), issued['grant'], permission='attachment.stage')
        assert 'attachment.stage' not in decode_room_grant(target.adapter._room_grant_secret(),
            issued['grant'], permission='status')['permissions']
    finally:
        process_ownership.release(target.home)


async def staged_batch(target):
    from dataclasses import replace
    from gateway.hosted_room_peer import attachment_manifest_digest, decode_room_grant
    from gateway.platforms.api_server_room_attachments import _default_spool, _write_guard
    from tests.gateway.test_canonical_peer_target_setup import request
    from tests.gateway.test_canonical_peer_text_admission import dispatch
    from tui_gateway.hosted_room_peer_attachments import bound_attachment_payloads
    issued = await invite(target)
    store, event, attachments, raw = source_files(target.home)
    pending = bound_attachment_payloads(store, 'room-one', 'member-one', attachments)
    manifest = [{k: v for k, v in p.items() if k != 'data'} for p in pending]
    value = replace(dispatch(issued), attachment_manifest_digest=attachment_manifest_digest(manifest))
    claims = decode_room_grant(target.adapter._room_grant_secret(), issued['grant'], permission='attachment.stage')
    req = request({}, token=issued['grant'])
    spool = _default_spool()
    spool.prepare(value, manifest, authorize_write=_write_guard(target.adapter, req, claims, 'attachment.stage', value))
    return issued, value, claims, req, spool, manifest, raw


@pytest.mark.asyncio
@pytest.mark.parametrize('denial', ['shared-state.db', 'state.db', 'policy', 'epoch', 'owner'])
async def test_late_stage_denial_cannot_commit(files_target, denial):
    from gateway import hosted_rooms
    from gateway.platforms.api_server_room_attachments import _write_guard, RoomGrantReauthorizationRequired
    from hermes_state_runtime import begin_runtime_epoch
    issued, value, claims, req, spool, manifest, raw = await staged_batch(files_target)
    if denial.endswith('.db'):
        hosted_rooms.revoke_room_grant_id(files_target.home / denial, claims=claims, expires_at=claims['status_expires_at'])
    elif denial == 'policy':
        (files_target.home / 'config.yaml').write_text('agent:\n  max_turns: 3\napprovals:\n  mode: manual\n')
    elif denial == 'epoch':
        begin_runtime_epoch(files_target.db, instance_id='inert-new-epoch')
    else:
        files_target.runner.session_authorities._by_key.clear()
    with pytest.raises(RoomGrantReauthorizationRequired):
        spool.put(claims=claims, task_id=value.task_id, execution_generation=1,
            attachment_id=manifest[0]['attachment_id'], data=raw[0],
            authorize_write=_write_guard(files_target.adapter, req, claims, 'attachment.stage', value))
    with spool._transaction() as conn:
        assert conn.execute('SELECT sum(stored) FROM roomlink_attachment_files').fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['prepare', 'put', 'discard'])
async def test_held_shared_and_profile_connections_fence_spool_commit(files_target, monkeypatch, operation):
    import sqlite3
    from contextlib import contextmanager
    from gateway.platforms.api_server_room_attachments import _write_guard
    issued, value, claims, req, spool, manifest, raw = await staged_batch(files_target)
    transaction = spool._transaction
    held = []
    @contextmanager
    def at_commit(*, immediate=False):
        with transaction(immediate=immediate) as conn:
            yield conn
            if immediate:
                for name in ('shared-state.db', 'state.db'):
                    with sqlite3.connect(files_target.home / name, timeout=0) as other:
                        with pytest.raises(sqlite3.OperationalError, match='locked'):
                            other.execute('BEGIN IMMEDIATE')
                        held.append(name)
    monkeypatch.setattr(spool, 'prune', lambda **kwargs: 0)  # No expired fixture rows.
    monkeypatch.setattr(spool, '_transaction', at_commit)
    guard = _write_guard(files_target.adapter, req, claims, 'attachment.stage', value)
    if operation == 'prepare':
        spool.prepare(value, manifest, authorize_write=guard)
    elif operation == 'put':
        spool.put(claims=claims, task_id=value.task_id, execution_generation=1,
            attachment_id=manifest[0]['attachment_id'], data=raw[0], authorize_write=guard)
    else:
        spool.discard_attempt(claims=claims, task_id=value.task_id, execution_generation=1, authorize_write=guard)
    assert held == ['shared-state.db', 'state.db']


@pytest.mark.asyncio
@pytest.mark.parametrize('store', ['shared-state.db', 'state.db'])
async def test_failed_write_lock_preserves_preexisting_spool(files_target, monkeypatch, store):
    import sqlite3
    from gateway import hosted_rooms
    from gateway.platforms.api_server_room_attachments import _write_guard
    issued, value, claims, req, spool, manifest, raw = await staged_batch(files_target)
    connect = hosted_rooms._connect
    def impatient(path):
        conn = connect(path)
        conn.execute('PRAGMA busy_timeout=0')
        return conn
    monkeypatch.setattr(hosted_rooms, '_connect', impatient)
    original = spool._connect
    def impatient_spool():
        conn = original()
        conn.execute('PRAGMA busy_timeout=0')
        return conn
    monkeypatch.setattr(spool, '_connect', impatient_spool)
    monkeypatch.setattr(spool, 'prune', lambda **kwargs: 0)
    with sqlite3.connect(files_target.home / store) as lock:
        lock.execute('BEGIN IMMEDIATE')
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            spool.discard_attempt(claims=claims, task_id=value.task_id, execution_generation=1,
                authorize_write=_write_guard(files_target.adapter, req, claims, 'status', value))
        lock.rollback()
    with spool._transaction() as conn:
        assert conn.execute('SELECT count(*) FROM roomlink_attachment_files').fetchone()[0] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', ['manifest', 'bytes', 'generation', 'recipient'])
async def test_exact_attempt_rejects_conflicting_input(files_target, changed):
    from dataclasses import replace
    from gateway.platforms.api_server_room_attachments import RoomAttachmentSpoolError
    issued, value, claims, req, spool, manifest, raw = await staged_batch(files_target)
    if changed == 'manifest':
        with pytest.raises(RoomAttachmentSpoolError):
            spool.prepare(value, [manifest[0] | {'name': 'changed.txt'}, manifest[1]])
    else:
        with pytest.raises(RoomAttachmentSpoolError):
            spool.put(claims=claims | ({'member_id': 'foreign'} if changed == 'recipient' else {}),
                task_id=value.task_id, execution_generation=2 if changed == 'generation' else 1,
                attachment_id=manifest[0]['attachment_id'], data=raw[0] + (b'changed' if changed == 'bytes' else b''))
    with spool._transaction() as conn:
        assert conn.execute('SELECT sum(stored) FROM roomlink_attachment_files').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_expired_staging_status_live_can_dispose_without_terminal_proof(files_target, monkeypatch):
    import asyncio
    from gateway.hosted_room_peer import decode_room_grant, HostedRoomGrantError
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    issued, value, claims, req, spool, manifest, raw = await staged_batch(files_target)
    monkeypatch.setattr('gateway.hosted_room_peer.time.time', lambda: claims['expires_at'] + 1)
    with pytest.raises(HostedRoomGrantError):
        decode_room_grant(files_target.adapter._room_grant_secret(), issued['grant'], permission='attachment.stage')
    decode_room_grant(files_target.adapter._room_grant_secret(), issued['grant'], permission='status')
    inprocess_http(files_target, monkeypatch)
    client = PeerRunsHTTPClient(base_url='http://127.0.0.1:8642', api_key='')
    result = await asyncio.to_thread(client.discard_attachments, task_id=value.task_id,
        execution_generation=1, grant=issued['grant'])
    assert result['removed'] == 1
    assert files_target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('denial', ['grant', 'refs'])
async def test_atomic_custody_failure_rolls_back_admission_and_own_run(files_target, monkeypatch, denial):
    from gateway import session_api_turn, hosted_rooms
    from gateway.platforms import api_server_runs
    from tests.gateway.test_canonical_peer_text_admission import run_request
    issued, value, claims, req, spool, manifest, raw = await staged_batch(files_target)
    for item, data in zip(manifest, raw):
        spool.put(claims=claims, task_id=value.task_id, execution_generation=1,
                  attachment_id=item['attachment_id'], data=data)
    if denial == 'refs':
        files_target.db._conn.execute("CREATE TRIGGER deny_refs AFTER INSERT ON input_custody_refs BEGIN SELECT RAISE(ABORT, 'inert custody failure'); END")
    else:
        original = session_api_turn.admit_api_turn
        def interleave(*args, **kwargs):
            hosted_rooms.revoke_room_grant_id(files_target.home / 'state.db', claims=claims,
                                              expires_at=claims['status_expires_at'])
            return original(*args, **kwargs)
        monkeypatch.setattr(session_api_turn, 'admit_api_turn', interleave)
    async def forbidden(*args, **kwargs):
        pytest.fail('denied admission launched execution')
    monkeypatch.setattr(api_server_runs, '_execute_run', forbidden)
    response = await files_target.adapter._handle_runs(run_request(issued['grant'], value))
    assert response.status in (409, 503), response.text
    assert files_target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
    assert files_target.db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
    assert files_target.adapter._run_idempotency_store._conn.execute('SELECT count(*) FROM run_idempotency').fetchone()[0] == 0
    assert not files_target.adapter._active_run_tasks


@pytest.mark.asyncio
async def test_peer_content_rejects_unaccepted_or_changed_document(files_target, monkeypatch):
    from gateway.session_peer_input import peer_input_content
    from gateway.session_api_turn import admit_api_turn
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import RuntimeStoreError
    issued, value, claims, req, spool, manifest, raw = await staged_batch(files_target)
    for item, data in zip(manifest, raw):
        spool.put(claims=claims, task_id=value.task_id, execution_generation=1,
                  attachment_id=item['attachment_id'], data=data)
    session_id = await files_target.adapter._ensure_hosted_member_session(value)
    owner, ref, row = admit_api_turn(files_target.adapter, user_message=value.prompt, conversation_history=[],
        session_id=session_id, request_id='test-admit', room_dispatch=value.as_mapping(),
        room_execution_policy=issued['catalog']['execution_policy'], _room_grant_token=issued['grant'])
    import copy
    forged = copy.deepcopy(row['payload'])
    forged['api_turn_v1']['settings']['room_input_media']['request_id'] = 'not-accepted'
    with pytest.raises(RuntimeStoreError):
        peer_input_content(owner, ref, forged)
    from pathlib import Path
    Path(row['payload']['api_turn_v1']['settings']['room_input_media']['media'][0]['path']).write_bytes(b'corrupt')
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
        peer_input_content(owner, ref, row['payload'])
