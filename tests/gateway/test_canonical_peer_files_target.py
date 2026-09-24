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


def inprocess_http(target, monkeypatch, *, responses=None):
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
    native = target.adapter._http_route_table()
    # Mirror the real app router registration, including root profile aliases.
    table = native + [(verb, "/p/{profile}" + path, handler) for verb, path, handler in native]
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
        if "profile" in match:
            assert target.adapter._resolve_request_profile(req) is None
        req.json = AsyncMock(return_value=json.loads(data) if method == 'POST' and data else {})
        response = await handler(req)
        if responses is not None:
            responses.append((method, path, response.status, json.loads(response.body)))
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
    calls = inprocess_http(target, monkeypatch)
    client = PeerRunsHTTPClient(base_url='http://127.0.0.1:8642', api_key='')
    discovered = await asyncio.to_thread(client.probe, grant=issued['grant'])
    catalog = discovered['catalog']
    assert catalog == issued['catalog']
    from gateway.hosted_room_peer import decode_room_grant
    claims = decode_room_grant(target.adapter._room_grant_secret(), issued['grant'], permission='attachment.stage')
    assert claims['execution_policy_digest'] == catalog['execution_policy']['policy_digest']
    assert set(claims['permissions']) == {'approve', 'dispatch', 'status', 'stop', 'attachment.stage'}
    route = PeerMemberRoute(home_install_id='home-install', member_id='member-one',
        target_install_id=catalog['installation_id'], target_profile='default',
        capability_digest=catalog['catalog_digest'], cancellation_scope_id='cancel-one', trace_id='trace-one',
        grant=issued['grant'], execution_policy_digest=catalog['execution_policy']['policy_digest'],
        attachments=catalog['attachments'])
    transport = PeerHostedRoomTransport(binding=HostedRoomBinding('room-one', 'home-gateway', 1),
        route=route, client=client, source_event_seq=event['seq'], task_id='task-one',
        execution_generation=1, attachment_store=store)
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
    from gateway.platforms.api_server_room_attachments import _request_spool
    spool = _request_spool()
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
    if denial == 'uninitialized':
        process_ownership.reserve([target.home])
    try:
        assert process_ownership.owns(target.home) is (denial == 'uninitialized')
        _, catalog = _local_room_catalog(target.adapter, 'other' if denial == 'named' else 'default',
                                         hosted_rooms.local_authority_gateway_id())
        assert catalog['attachments'] is False
        assert list(target.db._conn.execute("SELECT name FROM sqlite_master ORDER BY name")) == before
        assert target.db._conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 0
    finally:
        if denial == 'uninitialized':
            process_ownership.release(target.home)


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


async def staged_batch(target, *, prepare=True):
    from dataclasses import replace
    from gateway.hosted_room_peer import attachment_manifest_digest, decode_room_grant
    from gateway.platforms.api_server_room_attachments import _request_spool, _write_guard
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
    spool = _request_spool()
    if prepare:
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


async def http_admitted_files(target, monkeypatch):
    """Real source/event bytes, signed grant, HTTP batch and canonical admission."""
    import asyncio
    from types import SimpleNamespace
    from gateway.platforms import api_server_runs
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    issued, value, claims, req, spool, manifest, raw = await staged_batch(target, prepare=False)
    responses = []
    calls = inprocess_http(target, monkeypatch, responses=responses)
    client = PeerRunsHTTPClient(base_url='http://127.0.0.1:8642', api_key='')
    discovered = await asyncio.to_thread(client.probe, grant=issued['grant'])
    assert discovered['catalog'] == issued['catalog']
    assert discovered['catalog']['attachments'] is True
    launched = []
    async def inert(adapter, launch, **kwargs):
        launched.append(launch.admission)
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    staged = await asyncio.to_thread(client.stage_attachments, dispatch=value.as_mapping(),
        attachments=[dict(item, data=data) for item, data in zip(manifest, raw)], grant=issued['grant'])
    assert staged['complete'] is True
    accepted = await asyncio.to_thread(client.dispatch, dispatch=value.as_mapping(), grant=issued['grant'])
    await asyncio.sleep(0)
    assert len(launched) == 1
    assert calls.count(('POST', '/v1/room-members/attachments')) == 1
    assert len([call for call in calls if call[0] == 'PUT']) == 2
    return SimpleNamespace(issued=issued, value=value, claims=claims, req=req, spool=spool,
        manifest=manifest, raw=raw, responses=responses, calls=calls, client=client,
        launched=launched, accepted=accepted)


def files_state(target, spool):
    """Compare actual rows AND physical bytes, not only mutation counters."""
    import json
    from pathlib import Path
    from gateway.hosted_room_grant_state import grant_state_db_paths
    from gateway import hosted_rooms
    with spool._transaction() as conn:
        staging = {table: [tuple(row) for row in conn.execute(f'SELECT * FROM {table} ORDER BY 1,2')]
            for table in ('roomlink_attachment_batches', 'roomlink_attachment_files',
                          'roomlink_attachment_attempt_fences')}
    custody = [tuple(row) for row in target.db._conn.execute('SELECT * FROM input_custody_refs ORDER BY 1,2')]
    copies = [tuple(row) for row in target.db._conn.execute('SELECT * FROM input_custody_copies ORDER BY 1')]
    payloads = [json.loads(row[0]) for row in target.db._conn.execute('SELECT payload_json FROM session_admissions')]
    accepted_bytes = {r['path']: Path(r['path']).read_bytes() for p in payloads
        for r in p.get('api_turn_v1', {}).get('settings', {}).get('room_input_media', {}).get('media', [])}
    reservations = []
    for path in grant_state_db_paths(target.home):
        with hosted_rooms._transaction(path) as conn:
            reservations.append([tuple(row) for row in conn.execute('SELECT * FROM hosted_room_peer_reservations ORDER BY 1')])
    return dict(staging=staging, bytes={p.name: p.read_bytes() for p in spool.root.iterdir()},
        custody=custody, copies=copies, accepted_bytes=accepted_bytes, reservations=reservations)


@pytest.mark.asyncio
@pytest.mark.parametrize('metadata_only', [False, True], ids=['terminal-row', 'terminal-metadata'])
async def test_terminal_http_replay_after_spool_disposal(files_target, monkeypatch, metadata_only):
    import asyncio
    import json
    from pathlib import Path
    from hermes_state_runtime import claim_session_input, settle_session_input, RuntimeStoreError
    from hermes_state_terminal import ADMISSION_PREFIX, identity_key, terminal_admission
    from gateway.session_peer_input import prepare_peer_input
    target = files_target
    case = await http_admitted_files(target, monkeypatch)
    authority, ref, admitted = case.launched[0]
    refs = admitted['payload']['api_turn_v1']['settings']['room_input_media']['media']
    assert [Path(r['path']).read_bytes() for r in refs] == case.raw
    assert admitted['payload']['text'] == case.value.prompt
    # Data-only fixture transition: no executor, worker, Stop, adoption or retirement.
    started = claim_session_input(target.db, epoch=authority.epoch, session_id=ref.session_id)
    assert started['admission_id'] == admitted['admission_id']
    settled = settle_session_input(target.db, epoch=authority.epoch, admission_id=started['admission_id'],
        generation=started['generation'], outcome='completed',
        result={'result': {'final_response': 'synthetic terminal result', 'messages': []}, 'usage': {}})
    assert settled['status'] == 'terminal'
    terminal = target.adapter._set_run_status(case.accepted['run_id'], 'completed', output='synthetic terminal result')
    receipt = tuple(target.adapter._run_idempotency_store._conn.execute('SELECT * FROM run_idempotency').fetchone())
    original_digest = target.db._conn.execute('SELECT payload_digest FROM session_admissions').fetchone()[0]
    if metadata_only:
        # Exact payload-free schema of terminal receipts; preserve the ORIGINAL
        # admission identity/digest. Do not invoke session retirement/lifecycle.
        def trim(conn):
            saved = dict(conn.execute('SELECT * FROM session_admissions WHERE admission_id=?',
                (admitted['admission_id'],)).fetchone())
            saved.update(payload_json='{}', lineage_json='[]')
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                (ADMISSION_PREFIX + saved['admission_id'], json.dumps(saved)))
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                (identity_key('api', ref.session_id, saved['request_id']), json.dumps(saved['admission_id'])))
            conn.execute('DELETE FROM session_admissions WHERE admission_id=?', (saved['admission_id'],))
        target.db._execute_write(trim)
        saved = terminal_admission(target.db._conn, admitted['admission_id'])
        assert saved['payload_json'] == '{}' and saved['payload_digest'] == original_digest
    else:
        saved = dict(target.db._conn.execute('SELECT * FROM session_admissions').fetchone())
    retained = [tuple(r) for r in target.db._conn.execute('SELECT * FROM input_custody_refs ORDER BY ordinal')]
    await asyncio.to_thread(case.client.discard_attachments, task_id=case.value.task_id,
        execution_generation=1, grant=case.issued['grant'])
    assert not list(case.spool.root.iterdir())
    with case.spool._transaction() as conn:
        assert conn.execute('SELECT count(*) FROM roomlink_attachment_batches').fetchone()[0] == 0
    def forbidden(*args, **kwargs):
        pytest.fail('terminal replay recaptured or read discarded staging')
    monkeypatch.setattr(case.spool, 'materialize', forbidden)
    monkeypatch.setattr('gateway.session_peer_input.capture_native_media', forbidden)
    monkeypatch.setattr('gateway.hosted_room_input_preparation.prepare_verified_documents', forbidden)
    # Evict only the synthetic in-memory status; the HTTP receipt remains durable.
    target.adapter._run_statuses.pop(case.accepted['run_id'])
    replay = await asyncio.to_thread(case.client.dispatch, dispatch=case.value.as_mapping(), grant=case.issued['grant'])
    assert replay == case.accepted | {'replayed': True}
    run_responses = [body for method, path, status, body in case.responses if path == '/v1/runs' and status == 202]
    assert run_responses[-1] == {'run_id': case.accepted['run_id'], 'status': 'completed', 'replayed': True}
    assert terminal['status'] == 'completed' and len(case.launched) == 1
    assert tuple(target.adapter._run_idempotency_store._conn.execute('SELECT * FROM run_idempotency').fetchone()) == receipt
    assert [tuple(r) for r in target.db._conn.execute('SELECT * FROM input_custody_refs ORDER BY ordinal')] == retained
    assert [Path(r['path']).read_bytes() for r in refs] == case.raw
    assert target.db._conn.execute('SELECT count(*) FROM worker_executions').fetchone()[0] == 0
    if metadata_only:
        assert terminal_admission(target.db._conn, admitted['admission_id']) == saved
        assert target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
        # A direct helper has no durable HTTP fingerprint to recover the erased
        # payload. It must refuse structurally, without recapture or KeyError.
        with pytest.raises(RuntimeStoreError, match='storage_unavailable|admission_conflict'):
            prepare_peer_input(authority, session_id=ref.session_id, request_id=saved['request_id'],
                dispatch=case.value, payload=admitted['payload'])
    else:
        assert dict(target.db._conn.execute('SELECT * FROM session_admissions').fetchone()) == saved


@pytest.mark.asyncio
@pytest.mark.parametrize('rights', [('status',), ('approve', 'dispatch', 'status', 'stop')],
                         ids=['status-only', 'old-four-rights'])
async def test_status_only_http_cleanup_cannot_dispose_uploaded_files(files_target, monkeypatch, rights):
    import asyncio
    from gateway.hosted_room_peer import issue_room_grant, decode_room_grant
    from gateway import hosted_rooms
    from gateway.hosted_room_grant_state import grant_state_db_paths
    from tests.gateway.test_canonical_peer_target_setup import request
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    target = files_target
    case = await http_admitted_files(target, monkeypatch)
    signer = {k: case.claims[k] for k in ('grant_id', 'room_id', 'home_install_id',
        'authority_gateway_id', 'authority_epoch', 'member_id', 'target_install_id',
        'target_profile', 'execution_policy_digest', 'issued_at')}
    restricted = issue_room_grant(target.adapter._room_grant_secret(), **signer, permissions=rights,
        ttl_seconds=case.claims['expires_at'] - case.claims['issued_at'], status_expires_at=case.claims['status_expires_at'])
    claims = decode_room_grant(target.adapter._room_grant_secret(), restricted, permission='status')
    assert set(claims['permissions']) == set(rights)
    assert {k: claims[k] for k in signer} == signer
    assert target.adapter._room_grant_claims(request({}, token=restricted), permission='status') == claims
    assert all(hosted_rooms.peer_room_grant_is_current(p, claims=claims) for p in grant_state_db_paths(target.home))
    before = files_state(target, case.spool)
    assert len(before['bytes']) == 2 and len(before['custody']) == 2
    with pytest.raises(PeerRunsHTTPError) as denied:
        await asyncio.to_thread(case.client.discard_attachments, task_id=case.value.task_id,
            execution_generation=1, grant=restricted)
    assert denied.value.status_code == 401
    assert denied.value.error_code == 'invalid_room_grant'
    assert 'does not permit attachment cleanup' in case.responses[-1][3]['error']['message']
    assert files_state(target, case.spool) == before
    assert case.responses[-1][:3] == ('DELETE', f'/v1/room-members/attachments/{case.value.task_id}/1', 401)


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['prepare', 'put', 'discard'])
async def test_http_partial_shared_revoke_failure_fences_files_write(files_target, monkeypatch, operation):
    import asyncio
    import sqlite3
    from gateway import hosted_rooms
    from gateway.hosted_room_grant_state import grant_state_db_paths, revoke_grant_state
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    target = files_target
    case = await http_admitted_files(target, monkeypatch)
    from dataclasses import replace
    candidate = replace(case.value, task_id='unaccepted-files')
    if operation == 'put':
        await asyncio.to_thread(case.client._request, '/v1/room-members/attachments', method='POST',
            body={'hosted_room_dispatch': candidate.as_mapping(), 'attachments': case.manifest},
            room_grant=case.issued['grant'], reject_redirects=True)
        await asyncio.to_thread(case.client._put_attachment,
            f'/v1/room-members/attachments/{candidate.task_id}/1/{case.manifest[0]["attachment_id"]}',
            data=case.raw[0], grant=case.issued['grant'])
    paths = grant_state_db_paths(target.home)
    # Actual SQLite write failure on the shared leg, not a mocked revoke result.
    with hosted_rooms._transaction(paths[0], immediate=True) as conn:
        conn.execute("CREATE TRIGGER deny_shared_revoke BEFORE INSERT ON hosted_room_revoked_grants BEGIN SELECT RAISE(ABORT, 'synthetic shared revoke failure'); END")
    before = files_state(target, case.spool)
    initial = target.adapter._room_grant_claims
    preauthorized = []
    def interleave(request, *, permission):
        claims = initial(request, permission=permission)
        preauthorized.append(permission)
        with pytest.raises(sqlite3.IntegrityError, match='synthetic shared revoke failure'):
            revoke_grant_state(paths, claims=claims, expires_at=claims['status_expires_at'])
        # Verify both legs directly: failed shared commit, successful local deny.
        with hosted_rooms._transaction(paths[0]) as shared, hosted_rooms._transaction(paths[1]) as profile:
            assert shared.execute('SELECT count(*) FROM hosted_room_revoked_grants').fetchone()[0] == 0
            assert profile.execute('SELECT count(*) FROM hosted_room_revoked_grants').fetchone()[0] == 1
            assert shared.execute('SELECT revoked_at FROM hosted_room_peer_reservations').fetchone()[0] is None
            assert profile.execute('SELECT revoked_at FROM hosted_room_peer_reservations').fetchone()[0] is not None
        assert not hosted_rooms.room_grant_is_revoked(paths[0], claims=claims)
        assert hosted_rooms.room_grant_is_revoked(paths[1], claims=claims)
        return claims
    monkeypatch.setattr(target.adapter, '_room_grant_claims', interleave)
    from gateway import session_peer_target
    require_current = session_peer_target.require_current_grant
    held = []
    def inspect_held(conn, claims):
        from pathlib import Path
        assert conn.in_transaction
        held.append(Path(conn.execute('PRAGMA database_list').fetchone()[2]))
        return require_current(conn, claims)
    monkeypatch.setattr(session_peer_target, 'require_current_grant', inspect_held)
    with pytest.raises(PeerRunsHTTPError) as denied:
        if operation == 'prepare':
            await asyncio.to_thread(case.client.stage_attachments, dispatch=candidate.as_mapping(),
                attachments=[dict(item, data=data) for item, data in zip(case.manifest, case.raw)], grant=case.issued['grant'])
        elif operation == 'put':
            await asyncio.to_thread(case.client._put_attachment,
                f'/v1/room-members/attachments/{candidate.task_id}/1/{case.manifest[1]["attachment_id"]}',
                data=case.raw[1], grant=case.issued['grant'])
        else:
            await asyncio.to_thread(case.client.discard_attachments, task_id=case.value.task_id,
                execution_generation=1, grant=case.issued['grant'])
    assert preauthorized == ['status' if operation == 'discard' else 'attachment.stage']
    assert held == list(paths)
    assert denied.value.status_code == 401 and denied.value.error_code == 'invalid_room_grant'
    after = files_state(target, case.spool)
    assert after['reservations'][0] == before['reservations'][0]
    assert after['reservations'][1] != before['reservations'][1]
    assert {k: v for k, v in after.items() if k != 'reservations'} == {k: v for k, v in before.items() if k != 'reservations'}
    assert len(case.launched) == 1


@pytest.mark.asyncio
async def test_native_files_catalog_consumed_by_outbound_registration(files_target, monkeypatch):
    from types import SimpleNamespace
    from gateway import hosted_rooms, hosted_room_links
    from gateway.hosted_room_peer import decode_room_grant
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from tui_gateway import hosted_room_service
    from tests.gateway.test_canonical_peer_target_setup import invitation
    target = files_target
    # Reuse native registration fixture's execution-only seam, not a new principal.
    monkeypatch.setattr(hosted_room_service, 'HostedRoomRuntime', lambda **kwargs: SimpleNamespace(
        status=lambda: {'running': True, 'stopping': False}, wakeup=lambda: None))
    config = target.home / 'config.yaml'
    config.write_text(config.read_text() + '\ngateway:\n  room_link_url: https://target.example.test\n')
    service = CanonicalHostedRoomService(target.authority, None)
    target.authority.hosted_room_service = service
    installation = hosted_rooms.local_authority_gateway_id()
    issued = await invite(target, invitation() | {'home_install_id': installation, 'authority_gateway_id': installation})
    claims = decode_room_grant(target.adapter._room_grant_secret(), issued['grant'], permission='attachment.stage')
    assert set(claims['permissions']) == {'approve', 'dispatch', 'status', 'stop', 'attachment.stage'}
    service.authorize_room(target.connection.actor.subject, 'room-one', create=True)
    hosted_rooms.create_room(target.db.db_path, room_id='room-one', name='Room', authority_gateway_id=installation,
        members=[{'member_id': 'member-one', 'profile': 'default', 'handle': 'member-one', 'target': {
            'kind': 'peer', 'peer_id': installation, 'installation_id': installation, 'profile': 'default',
            'capability_digest': issued['catalog']['catalog_digest']}}])
    responses = []
    calls = inprocess_http(target, monkeypatch, responses=responses)
    result = await target.connection.dispatch(dict(id=2, method='groups.peer.register', params=dict(
        request_id='register-files', room_id='room-one', member_id='member-one', target_url=issued['endpoint']['url'],
        target_profile='default', grant=issued['grant'], catalog=issued['catalog'],
        cancellation_scope_id='cancel-one', trace_id='trace-one')))
    assert result.get('result', {}).get('registered') is True, (result, calls, responses)
    assert ('GET', '/p/default/v1/room-members/capabilities') in calls
    assert all(body['catalog'] == issued['catalog'] for _, path, status, body in responses
        if path == '/p/default/v1/room-members/capabilities' and status == 200)
    stored = hosted_room_links.load_room_link(target.db.db_path, room_id='room-one', member_id='member-one')
    assert stored.grant == issued['grant'] and stored.catalog.as_mapping() == issued['catalog']
    bound = service.peer_routes[('room-one', 'member-one')]
    assert bound.attachments is True and bound.grant == issued['grant']
    assert bound.capability_digest == issued['catalog']['catalog_digest']
    assert bound.execution_policy_digest == claims['execution_policy_digest']


@pytest.mark.asyncio
async def test_prior_http_admission_then_upload_413_is_not_terminal(files_target, monkeypatch):
    import asyncio
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from gateway.hosted_room_driver import TaskIdentity
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    from tui_gateway.hosted_room_peer_transport import PeerMemberRoute, PeerHostedRoomTransport
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    case = await http_admitted_files(files_target, monkeypatch)
    value = case.value
    route = PeerMemberRoute(**{k: getattr(value, k) for k in ('home_install_id', 'member_id',
        'target_install_id', 'target_profile', 'capability_digest', 'execution_policy_digest',
        'cancellation_scope_id', 'trace_id')}, grant=case.issued['grant'], attachments=True)
    transport = PeerHostedRoomTransport(binding=HostedRoomBinding(value.room_id, value.authority_gateway_id, 1),
        route=route, client=case.client, source_event_seq=value.source_event_seq, task_id=value.task_id,
        execution_generation=1, attachment_store=HostedRoomAttachmentStore(files_target.home / 'source.db'))
    with case.spool._transaction() as conn:
        prepared_at = conn.execute('SELECT created_at FROM roomlink_attachment_batches').fetchone()[0]
    monkeypatch.setattr(case.spool, 'clock', lambda: prepared_at)
    before = files_state(files_target, case.spool)
    admissions = [tuple(row) for row in files_target.db._conn.execute('SELECT * FROM session_admissions')]
    # Actual receiver rejects this upload; no fabricated client exception or grant.
    monkeypatch.setattr('gateway.platforms.api_server_room_attachments.MAX_ROOM_LINK_ATTACHMENT_BYTES', 1)
    terminal = []
    with pytest.raises(PeerRunsHTTPError) as denied:
        await asyncio.to_thread(transport.submit, profile='default', session_id=case.accepted['session_id'],
            prompt=value.prompt, source='bot_room', task=TaskIdentity(value.room_id, value.task_id, 'thread-one', 'turn-one'),
            execution_generation=1, on_terminal=terminal.append,
            attachments=[{k: v for k, v in item.items() if k != 'sha256'} | {'event_id': 'event-one'}
                         for item in case.manifest])
    assert transport._dispatch == value
    assert denied.value.status_code == 413 and denied.value.retryable is False
    assert denied.value.not_admitted is False and denied.value.dispatch_not_attempted is True
    assert terminal == [] and not any(method == 'DELETE' for method, _ in case.calls)
    assert files_state(files_target, case.spool) == before
    assert [tuple(row) for row in files_target.db._conn.execute('SELECT * FROM session_admissions')] == admissions
    assert len(case.launched) == 1
    assert case.calls.count(('POST', '/v1/runs')) == 1
    assert case.responses[-1][2] == 413
