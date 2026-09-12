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
        return {'owner': 'room-owner', 'target_home': authority.profile_id, 'prompt': 'input', 'attachments': []}
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


def test_source_attestation_binds_bytes_to_task_member_and_current_home(tmp_path, monkeypatch):
    import base64
    import time
    from types import SimpleNamespace
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import create_room, local_authority_gateway_id
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, RuntimeStoreError
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    import gateway.run
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    homes = {'other': str(tmp_path / 'profiles' / 'other')}
    monkeypatch.setattr(gateway.run, '_load_gateway_config', lambda: {'hosted_rooms': {'profiles': homes}})
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
        service = CanonicalHostedRoomService(authority, None)
        service.authorize_room('alice', 'room', create=True)
        gateway = local_authority_gateway_id()
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=gateway, members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
        data = b'owned source bytes' * 4000
        store = HostedRoomAttachmentStore(db.db_path)
        saved = store.put(room_id='room', upload_id='upload', kind='file', name='note.txt', mime='text/plain', data=data)
        manifest = [{k: saved[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
        store.commit_message(room_id='room', event_id='event', manifest=manifest, recipient_member_ids=['two'])
        bound = [{**manifest[0], 'event_id': 'event'}]
        identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
        payload = {'target_profile': 'other', 'target_member_id': 'two', 'source_event_seq': 1,
                   'prompt': 'frozen', 'attachments': bound}
        tasks.admit_task(db.db_path, identity, payload=payload, clock=time.time)
        lease = tasks.acquire_lease(db.db_path, room_id='room', gateway_id=gateway, authority_epoch=1,
                                   process_generation='test', ttl_seconds=30, clock=time.time)
        tasks.start_task(db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
        task, = tasks.list_tasks(db.db_path, room_id='room')
        selector = dict(room_id='room', member_id='two', profile='other')
        params = dict(task=asdict(identity), execution_generation=task['execution_generation'],
                      prompt='frozen', attachments=bound, _target_home=homes['other'])
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            service.attest(selector, 'submit', {**params, 'attachments': []})
        assert service.attest(selector, 'submit', params)['attachments'] == bound
        chunks = []
        for offset in range(0, len(data), 24576):
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
            transport_binding = {'source_home': str(tmp_path), 'target_home': homes['other'], 'selector': selector}
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
