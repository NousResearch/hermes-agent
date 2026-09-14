"""Real private owner sockets; no public identity or foreign database access."""
import asyncio
import json
import threading
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_hosted_rpc import owner  # noqa: F401


def _server(home):
    from gateway.control_socket import GatewayControlServer
    descriptor = {'runtime_protocol': 1, 'state': 'ready', 'instance_id': 'test',
                  'authority_epoch': 1, 'served_profiles': [{'home': str(home), 'profile_id': str(home)}],
                  'capabilities': ['session-authority-v1'], 'api_origin': 'http://127.0.0.1:1',
                  'supervisor': 'none'}
    return GatewayControlServer(home, verb_handlers={'identify': lambda: descriptor})


def test_authenticated_owner_transport_rechecks_source_and_cold_binding(owner, tmp_path):
    from gateway.session_hosted_transport import (
        HostedRoomOwnerRPC, install_hosted_transport, check_remote_hosted_admission,
        owner_request,
    )
    from gateway.hosted_room_driver import TaskIdentity
    from hermes_state_runtime import list_session_admissions, RuntimeStoreError
    authority, loop, _, _ = owner
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir(mode=0o700)
    target.mkdir(mode=0o700)
    authority.profile_id = str(target)
    source_authority = SimpleNamespace(profile_id=str(source))
    allowed = [True]
    task = TaskIdentity('room', 'task', 'thread', 'turn')
    def attest(selector, operation, params):
        if not allowed[0] or selector != dict(room_id='room', member_id='member', profile='default'):
            raise RuntimeStoreError('permission_denied')
        if operation in {'submit', 'execute'}:
            assert params['task'] == asdict(task)
            assert params['execution_generation'] == 1
            if operation == 'submit' and params['prompt'] != 'input':
                raise RuntimeStoreError('permission_denied')
        return {'owner': 'room-owner', 'target_home': authority.profile_id, 'prompt': 'input',
                'attachments': [], 'attachment_digests': []}
    servers = [_server(source), _server(target)]
    install_hosted_transport(servers[0], source_authority, loop, attest=attest)
    install_hosted_transport(servers[1], authority, loop, attest=lambda *a: None)
    for server in servers:
        assert asyncio.run_coroutine_threadsafe(server.start(), loop).result()
    try:
        rpc = HostedRoomOwnerRPC(home=target, source_home=source, room_id='room', member_id='member', profile='default')
        coords = dict(profile='default', source='bot_room')
        sid = rpc.create(**coords, title='Group: room')['session_id']
        args = dict(**coords, session_id=sid, prompt='input', task=task, execution_generation=1, on_terminal=lambda r: None)
        receipt = rpc.submit(**args)
        assert rpc.submit(**args)['admission_id'] == receipt['admission_id']
        rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        assert len(rows) == 1
        from gateway.session_contract import SessionRef
        ref = SessionRef(authority.profile_id, sid)
        assert check_remote_hosted_admission(authority, ref, rows[0]) is True
        # Recreate server routing to lose all process-local producer caches.
        install_hosted_transport(servers[1], authority, loop, attest=lambda *a: None)
        assert check_remote_hosted_admission(authority, ref, rows[0]) is True
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            rpc.submit(**{**args, 'prompt': 'forged'})
        raw = json.dumps({'protocol': 1, 'verb': 'hosted-producer', 'params': {}}).encode()
        assert not json.loads(servers[1].handle_request_line(raw))['ok']
        old_attest = servers[0].private_handlers['hosted-attest']
        def remapped(params, peer):
            return {**old_attest(params, peer), 'target_home': str(tmp_path / 'different')}
        servers[0].private_handlers['hosted-attest'] = remapped
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            rpc.history(**coords, session_id=sid)
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            check_remote_hosted_admission(authority, ref, rows[0])
        servers[0].private_handlers['hosted-attest'] = old_attest
        allowed[0] = False
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            check_remote_hosted_admission(authority, ref, rows[0])
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            rpc.history(**coords, session_id=sid)
        assert rows == list_session_admissions(authority.db, session_id=sid, pending_only=False)
    finally:
        for server in servers:
            asyncio.run_coroutine_threadsafe(server.stop(), loop).result()


class _SourceTask:
    """One committed attachment bound to a running task on a real source owner."""

    def __init__(self, tmp_path, monkeypatch, data, *, name='note.txt', mime='text/plain', kind='file'):
        import time
        from types import SimpleNamespace
        from gateway.session_hosted_service import CanonicalHostedRoomService
        from gateway import hosted_room_driver as tasks
        from gateway.hosted_rooms import create_room, local_authority_gateway_id
        from gateway.hosted_room_attachments import HostedRoomAttachmentStore
        from hermes_state import SessionDB
        from hermes_state_runtime import begin_runtime_epoch
        import gateway.run
        monkeypatch.setenv('HERMES_HOME', str(tmp_path))
        self.homes = {'other': str(tmp_path / 'profiles' / 'other')}
        monkeypatch.setattr(gateway.run, '_load_gateway_config', lambda: {'hosted_rooms': {'profiles': self.homes}})
        self.db = db = SessionDB(tmp_path / 'state.db')
        self.authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
        self.service = CanonicalHostedRoomService(self.authority, None)
        self.service.authorize_room('alice', 'room', create=True)
        self.gateway = local_authority_gateway_id()
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=self.gateway, members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
        self.data = data
        self.store = store = HostedRoomAttachmentStore(db.db_path)
        saved = store.put(room_id='room', upload_id='upload', kind=kind, name=name, mime=mime, data=data)
        manifest = [{k: saved[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
        store.commit_message(room_id='room', event_id='event', manifest=manifest, recipient_member_ids=['two'])
        self.bound = [{**manifest[0], 'event_id': 'event'}]
        self.identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
        payload = {'target_profile': 'other', 'target_member_id': 'two', 'source_event_seq': 1,
                   'prompt': 'frozen', 'attachments': self.bound}
        tasks.admit_task(db.db_path, self.identity, payload=payload, clock=time.time)
        lease = tasks.acquire_lease(db.db_path, room_id='room', gateway_id=self.gateway, authority_epoch=1,
                                    process_generation='test', ttl_seconds=30, clock=time.time)
        tasks.start_task(db.db_path, self.identity, lease, expected_cancel_generation=0, clock=time.time)
        self.task, = tasks.list_tasks(db.db_path, room_id='room')
        self.selector = dict(room_id='room', member_id='two', profile='other')
        self.params = dict(task=asdict(self.identity), execution_generation=self.task['execution_generation'],
                           prompt='frozen', attachments=self.bound, _target_home=self.homes['other'])
        self.binding = {'source_home': str(tmp_path), 'target_home': self.homes['other'], 'selector': self.selector}


def test_source_attestation_binds_bytes_to_task_member_and_current_home(tmp_path, monkeypatch):
    import base64
    from hermes_state_runtime import RuntimeStoreError
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    from gateway.session_hosted_transport import _CHUNK_BYTES
    source = _SourceTask(tmp_path, monkeypatch, b'owned source bytes' * 4000)
    with source.db:
        service, data, bound, task, gateway, homes, authority = (
            source.service, source.data, source.bound, source.task, source.gateway, source.homes, source.authority)
        selector, params = source.selector, source.params
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            service.attest(selector, 'submit', {**params, 'attachments': []})
        assert service.attest(selector, 'submit', params)['attachments'] == bound
        chunks = []
        for offset in range(0, len(data), _CHUNK_BYTES):
            result = service.attest(selector, 'attachment', {**params, 'index': 0, 'offset': offset})
            chunks.append(base64.b64decode(result['data_base64']))
        assert b''.join(chunks) == data
        from gateway.session_hosted_transport import install_hosted_transport, _attachment_data
        # The same chunk protocol crosses a real private socket; the target never
        # receives a filesystem path or opens the source database.
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever)
        thread.start()
        server = _server(tmp_path)
        install_hosted_transport(server, authority, loop, attest=service.attest)
        assert asyncio.run_coroutine_threadsafe(server.start(), loop).result()
        try:
            transport_binding = source.binding
            attested = service.attest(selector, 'submit', params)
            assert _attachment_data(transport_binding, attested, params) == [(bound[0], data)]
        finally:
            asyncio.run_coroutine_threadsafe(server.stop(), loop).result()
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            loop.close()
        for bad in ({'execution_generation': 999}, {'_target_home': str(tmp_path / 'wrong')},
                    {'attachments': [{**bound[0], 'event_id': 'other'}]}, {'index': 1}, {'offset': -1}):
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                service.attest(selector, 'attachment', {**params, 'index': 0, 'offset': 0, **bad})
        binding = HostedRoomBinding('room', gateway, 1)
        old_rpc = service._resolve_member_transport(binding, task)
        homes['other'] = str(tmp_path / 'replacement' / 'profiles' / 'other')
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            service.attest(selector, 'history', params)
        assert service._resolve_member_transport(binding, task) is not old_rpc
        homes.clear()
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            service._resolve_member_transport(binding, task)


def test_source_chunk_reads_only_the_requested_slice(tmp_path, monkeypatch):
    """Source cost per chunk is bounded by the chunk, not by the whole blob; the
    target still refuses a blob whose bytes drifted from the attested digest."""
    import base64
    import hashlib
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from gateway import session_hosted_transport as transport
    from hermes_state_runtime import RuntimeStoreError
    source = _SourceTask(tmp_path, monkeypatch, bytes(range(256)) * (4 * transport._CHUNK_BYTES // 256 + 1),
                         name='blob.bin', mime='application/octet-stream')
    with source.db:
        service, params = source.service, source.params
        monkeypatch.setattr(HostedRoomAttachmentStore, '_read_blob',
                            lambda *a, **k: pytest.fail('chunk request read the whole blob'))
        hashed = []
        real_sha256 = hashlib.sha256
        monkeypatch.setattr(hashlib, 'sha256', lambda data=b'': (hashed.append(len(data)), real_sha256(data))[1])
        result = service.attest(source.selector, 'attachment',
                                {**params, 'index': 0, 'offset': transport._CHUNK_BYTES})
        chunk = base64.b64decode(result['data_base64'])
        assert chunk == source.data[transport._CHUNK_BYTES:2 * transport._CHUNK_BYTES]
        assert result['sha256'] == real_sha256(source.data).hexdigest()
        assert sum(hashed) < len(chunk), 'source re-hashed more than the served slice'
        # End-to-end: the target verifies the assembled bytes against the attested digest.
        monkeypatch.setattr(transport, 'owner_request', lambda home, verb, p, **kw: service.attest(
            p['selector'], p['operation'], p['params']))
        attested = service.attest(source.selector, 'submit', params)
        assert transport._attachment_data(source.binding, attested, params) == [(source.bound[0], source.data)]
        blob, = source.store.blob_root.iterdir()
        blob.write_bytes(source.data[:-1] + bytes([source.data[-1] ^ 1]))
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            transport._attachment_data(source.binding, attested, params)


def test_attachment_chunks_fill_the_response_line_without_overflowing_it(tmp_path, monkeypatch):
    """Chunks nearly fill the single 512 KiB response line shared by the POSIX socket and
    the Windows pipe, never overflow it even with a long owner subject, and the transfer
    spends round-trips proportional to size / chunk."""
    import math
    from gateway.control_socket import _MAX_RESPONSE_BYTES
    from gateway import session_hosted_transport as transport
    from gateway.session_hosted_transport import _CHUNK_BYTES
    size = 2 * _CHUNK_BYTES + 1
    source = _SourceTask(tmp_path, monkeypatch, bytes(range(256)) * (size // 256 + 1),
                         name='blob.bin', mime='application/octet-stream')
    with source.db:
        service, params = source.service, source.params
        monkeypatch.setattr(service, '_owner', lambda room_id: 'o' * 4096)
        lines = []
        server = _server(tmp_path)
        real_handle = server.handle_request_line
        def handle(raw, *args):
            response = real_handle(raw, *args)
            if b'"hosted-attest"' in raw:
                lines.append(len(response))
            return response
        server.handle_request_line = handle
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever)
        thread.start()
        transport.install_hosted_transport(server, source.authority, loop, attest=service.attest)
        assert asyncio.run_coroutine_threadsafe(server.start(), loop).result()
        try:
            attested = service.attest(source.selector, 'submit', params)
            assert transport._attachment_data(source.binding, attested, params) == [(source.bound[0], source.data)]
        finally:
            asyncio.run_coroutine_threadsafe(server.stop(), loop).result()
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            loop.close()
        assert len(lines) == math.ceil(len(source.data) / _CHUNK_BYTES) == 3
        assert max(lines) <= _MAX_RESPONSE_BYTES
        assert max(lines) > _MAX_RESPONSE_BYTES * 3 // 4, 'chunks leave most of the response line unused'


def test_preflight_verifies_by_attested_digest_and_refuses_changed_source_bytes(owner, tmp_path, monkeypatch):
    """check_remote_hosted_admission proves the durable row still matches the source's
    bound input from source-attested digests, transferring no bytes; a source attachment
    re-pointed at different bytes (same id, name and size) is still refused."""
    from gateway import session_hosted_transport as transport
    from gateway.session_hosted_transport import HostedRoomOwnerRPC, check_remote_hosted_admission
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import list_session_admissions, RuntimeStoreError
    authority, loop, _, _ = owner
    target_home = tmp_path / 'profiles' / 'other'
    target_home.mkdir(parents=True, mode=0o700)
    authority.profile_id = str(target_home)
    source = _SourceTask(tmp_path, monkeypatch, b'document bytes ' * 20000)
    operations = []
    real_request = transport.owner_request
    def counting_request(home, verb, params, **kwargs):
        operations.append(params.get('operation'))
        return real_request(home, verb, params, **kwargs)
    monkeypatch.setattr(transport, 'owner_request', counting_request)
    servers = [_server(tmp_path), _server(target_home)]
    with source.db:
        transport.install_hosted_transport(servers[0], source.authority, loop, attest=source.service.attest)
        transport.install_hosted_transport(servers[1], authority, loop, attest=lambda *a: None)
        for server in servers:
            assert asyncio.run_coroutine_threadsafe(server.start(), loop).result()
        try:
            rpc = HostedRoomOwnerRPC(home=target_home, source_home=tmp_path, **source.selector)
            coords = dict(profile='other', source='bot_room')
            sid = rpc.create(**coords, title='Group: room')['session_id']
            rpc.submit(**coords, session_id=sid, prompt='frozen', task=source.identity,
                       execution_generation=1, attachments=source.bound, on_terminal=lambda r: None)
            row, = list_session_admissions(authority.db, session_id=sid, pending_only=False)
            assert 'attachment' in operations
            # Quiesce the driver-side history poller before counting the preflight.
            with rpc._lock:
                rpc.callbacks.clear()
            rpc._monitor.join(5)
            operations.clear()
            ref = SessionRef(authority.profile_id, sid)
            assert check_remote_hosted_admission(authority, ref, row) is True
            assert operations == ['execute'], 'preflight must verify by digest, not re-transfer bytes'
            # Same name and size, different bytes: re-point the committed row at another blob.
            changed = source.store.put(room_id='room', upload_id='upload-2', kind='file', name='note.txt',
                                       mime='text/plain', data=b'DOCUMENT BYTES ' * 20000)
            assert changed['size'] == source.bound[0]['size'] and changed['sha256'] != source.store.read(
                room_id='room', attachment_id=source.bound[0]['attachment_id'], event_id='event',
                recipient_member_id='two').attachment['sha256']
            with source.store._transaction() as conn:
                conn.execute('UPDATE hosted_room_attachments SET sha256=?, blob_id=(SELECT blob_id FROM '
                             'hosted_room_attachments WHERE attachment_id=?) WHERE attachment_id=?',
                             (changed['sha256'], changed['attachment_id'], source.bound[0]['attachment_id']))
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                check_remote_hosted_admission(authority, ref, row)
        finally:
            with rpc._lock:
                rpc.callbacks.clear()
            for server in servers:
                asyncio.run_coroutine_threadsafe(server.stop(), loop).result()


def test_preflight_refuses_a_retained_document_corrupted_or_missing_at_the_destination(owner, tmp_path, monkeypatch):
    """Documents ride in the prompt as content-addressed paths, so execution never re-hashes
    them: the digest-only preflight must itself refuse ``storage_unavailable`` when the
    destination bytes no longer match the source-attested digest (same-size mutation) or
    are gone, still without transferring bytes from the source. Images are unaffected."""
    from gateway import session_hosted_transport as transport
    from gateway.session_hosted_transport import HostedRoomOwnerRPC, check_remote_hosted_admission
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import list_session_admissions, RuntimeStoreError
    authority, loop, _, _ = owner
    target_home = tmp_path / 'profiles' / 'other'
    target_home.mkdir(parents=True, mode=0o700)
    authority.profile_id = str(target_home)
    source = _SourceTask(tmp_path, monkeypatch, b'document bytes ' * 2000)
    operations = []
    real_request = transport.owner_request
    def counting_request(home, verb, params, **kwargs):
        operations.append(params.get('operation'))
        return real_request(home, verb, params, **kwargs)
    monkeypatch.setattr(transport, 'owner_request', counting_request)
    servers = [_server(tmp_path), _server(target_home)]
    with source.db:
        transport.install_hosted_transport(servers[0], source.authority, loop, attest=source.service.attest)
        transport.install_hosted_transport(servers[1], authority, loop, attest=lambda *a: None)
        for server in servers:
            assert asyncio.run_coroutine_threadsafe(server.start(), loop).result()
        try:
            rpc = HostedRoomOwnerRPC(home=target_home, source_home=tmp_path, **source.selector)
            coords = dict(profile='other', source='bot_room')
            sid = rpc.create(**coords, title='Group: room')['session_id']
            rpc.submit(**coords, session_id=sid, prompt='frozen', task=source.identity,
                       execution_generation=1, attachments=source.bound, on_terminal=lambda r: None)
            row, = list_session_admissions(authority.db, session_id=sid, pending_only=False)
            with rpc._lock:
                rpc.callbacks.clear()
            rpc._monitor.join(5)
            ref = SessionRef(authority.profile_id, sid)
            assert check_remote_hosted_admission(authority, ref, row) is True
            from pathlib import Path
            retained = Path(row['payload']['text'].split('[Shared attachment] file: ')[1].strip())
            assert retained.read_bytes() == source.data
            operations.clear()
            retained.write_bytes(bytes([source.data[0] ^ 1]) + source.data[1:])
            with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
                check_remote_hosted_admission(authority, ref, row)
            retained.unlink()
            with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
                check_remote_hosted_admission(authority, ref, row)
            assert operations == ['execute', 'execute'], 'destination verification must not re-transfer bytes'
            assert row == list_session_admissions(authority.db, session_id=sid, pending_only=False)[0]
        finally:
            with rpc._lock:
                rpc.callbacks.clear()
            for server in servers:
                asyncio.run_coroutine_threadsafe(server.stop(), loop).result()
