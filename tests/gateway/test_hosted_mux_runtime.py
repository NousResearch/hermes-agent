"""Hosted multiplexing through real owners and private sockets, without inference."""
import asyncio
import base64
import json
from pathlib import Path
import threading
import time

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


@pytest.fixture
def mux(tmp_path, monkeypatch, request):
    from gateway.control_socket import GatewayControlServer
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership

    root, homes = _reserve_homes(tmp_path, monkeypatch)
    for name, home in homes:
        config = {'model': {'default': 'fixture-' + name}, 'platform_toolsets': {'cli': []},
                  'hosted_rooms': {'profiles': {n: str(h) for n, h in homes if h != home}}}
        (home / 'config.yaml').write_text(json.dumps(config))
    process_ownership.reserve([home for _, home in homes])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    runner = _runner(root, homes)
    def call(coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(15)
    try:
        call(initialize_gateway_runtime(runner))
        for authority in runner.session_authorities:
            # This suite checks creation/admission, not model execution.
            monkeypatch.setattr(authority, '_schedule', lambda ref: None)
        descriptor = runner.session_runtime_descriptor
        descriptor.update(state=getattr(request, 'param', 'ready'),
                          capabilities=['session-authority-v1'], api_origin='http://127.0.0.1:1',
                          supervisor='none')
        server = runner.session_control_server = GatewayControlServer(
            root, verb_handlers={'identify': lambda: descriptor})
        assert call(server.start())
        yield runner, dict(homes), loop, call
    finally:
        for authority in getattr(runner, 'session_authorities', []):
            service = getattr(authority, 'hosted_room_service', None)
            if service is not None:
                assert service.stop(timeout=5)
            authority.db.close()
        server = getattr(runner, 'session_control_server', None)
        if server is not None:
            call(server.stop())
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        assert not thread.is_alive()
        loop.close()
        for _, home in homes:
            process_ownership.release(home)


def test_hosted_lifecycle_serves_and_stops_every_authority(mux):
    from gateway.run_runtime import recover_gateway_native_sessions
    from gateway.session_hosted_service import stop_hosted_service

    runner, homes, _, call = mux
    assert call(recover_gateway_native_sessions(runner)) == {}
    services = {a.profile_id: getattr(a, 'hosted_room_service', None)
                for a in runner.session_authorities}
    assert all(services.values()), 'each served profile needs its own hosted service'
    assert len({id(s) for s in services.values()}) == len(homes)
    assert all(s.runtime.status()['running'] for s in services.values())
    assert call(runner._ensure_hosted_room_worker()) is runner.session_authority.hosted_room_service
    call(recover_gateway_native_sessions(runner))
    assert all(a.hosted_room_service is services[a.profile_id] for a in runner.session_authorities)
    assert call(stop_hosted_service(runner))
    assert all(not s.runtime.status()['running'] for s in services.values())


def test_secondary_owner_transport_routes_both_directions_through_mux(mux):
    from gateway import hosted_rooms
    from gateway.session_authorities import owner_scope
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway.session_hosted_transport import (
        HostedRoomOwnerRPC, install_hosted_transport, check_remote_hosted_admission,
    )
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from gateway.session_ingress_media import restore_native_media
    from hermes_state_runtime import list_session_admissions, RuntimeStoreError

    runner, homes, loop, _ = mux
    # Isolate routing from startup: construct the real per-profile services directly.
    for authority in runner.session_authorities:
        with owner_scope(authority):
            service = CanonicalHostedRoomService(authority, loop)
            authority.hosted_room_service = service
            install_hosted_transport(runner.session_control_server, authority, loop, attest=service.attest)
    source = runner.session_authorities.require(homes['alpha'])
    target = runner.session_authorities.require(homes['beta'])
    service = source.hosted_room_service
    with owner_scope(source):
        service.authorize_room('alice', 'room', create=True)
        hosted_rooms.create_room(source.db.db_path, room_id='room', name='Room',
            authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
            members=[{'member_id': 'helper', 'profile': 'beta', 'handle': 'helper'}])
    rpc = HostedRoomOwnerRPC(home=homes['beta'], source_home=homes['alpha'],
                            room_id='room', member_id='helper', profile='beta')
    sid = rpc.create(profile='beta', source='bot_room', title='Group: room')['session_id']
    assert target.db.get_session(sid)['source'] == 'bot_room'
    assert source.db.get_session(sid) is None
    assert runner.session_authority.db.get_session(sid) is None
    assert rpc.resume(profile='beta', source='bot_room', session_id=sid)['session_id'] == sid
    assert rpc.ref.profile_id == str(homes['beta'])
    with pytest.raises(RuntimeStoreError, match='profile_mismatch'):
        HostedRoomOwnerRPC(home=homes['default'], source_home=homes['alpha'],
            room_id='room', member_id='helper', profile='beta').create(
                profile='beta', source='bot_room', title='Group: room')

    data = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=')
    store = HostedRoomAttachmentStore(source.db.db_path)
    saved = store.put(room_id='room', upload_id='upload', kind='image', name='image.png',
                      mime='image/png', data=data)
    manifest = [{k: saved[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
    store.commit_message(room_id='room', event_id='event', manifest=manifest,
                         recipient_member_ids=['helper'])
    manifest[0]['event_id'] = 'event'
    identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
    payload = {'target_profile': 'beta', 'target_member_id': 'helper', 'source_event_seq': 1,
               'prompt': 'input', 'attachments': manifest}
    tasks.admit_task(source.db.db_path, identity, payload=payload, clock=time.time)
    lease = tasks.acquire_lease(source.db.db_path, room_id='room',
        gateway_id=hosted_rooms.local_authority_gateway_id(), authority_epoch=1,
        process_generation='fixture', ttl_seconds=30, clock=time.time)
    tasks.start_task(source.db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
    try:
        receipt = rpc.submit(profile='beta', source='bot_room', session_id=sid, prompt='input',
            task=identity, execution_generation=1, attachments=manifest, on_terminal=lambda row: None)
        repeated = rpc.submit(profile='beta', source='bot_room', session_id=sid, prompt='input',
            task=identity, execution_generation=1, attachments=manifest, on_terminal=lambda row: None)
        assert repeated['admission_id'] == receipt['admission_id']
        row, = list_session_admissions(target.db, session_id=sid, pending_only=False)
        assert row['status'] == 'queued'
        assert check_remote_hosted_admission(target, rpc.ref, row)
        with owner_scope(target):
            paths = restore_native_media(row['payload']['attachments_v1']['media'])
        assert len(paths) == 1
        assert Path(paths[0]).is_relative_to(homes['beta'])
        assert Path(paths[0]).read_bytes() == data
    finally:
        with rpc._lock:
            rpc.callbacks.clear()
        if rpc._monitor is not None:
            rpc._monitor.join(5)
            assert not rpc._monitor.is_alive()
        # The queued admission and its running room-attempt row stay as evidence
        # in the disposable database; no executor ran or needs to be settled.


def test_private_rpc_builds_policy_in_destination_scope(mux):
    from gateway.session_contract import Principal
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.session_policy import policy_for_source

    runner, homes, loop, _ = mux
    authority = runner.session_authorities.require(homes['beta'])
    actor = Principal('alice', authority.profile_id,
                      frozenset({'session:create', 'session:read'}), 'private-owner')
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='helper',
                                profile='beta', principal=actor, authorize=lambda *args: True)
    sid = rpc.create(profile='beta', source='bot_room', title='Group: room')['session_id']
    policy = policy_for_source(runner, authority.sessions[sid].source)
    assert policy.model == 'fixture-beta'
    assert policy.config()['model']['default'] == 'fixture-beta'


def test_room_coordinator_submits_once_to_destination_authority(mux, monkeypatch):
    from gateway import hosted_rooms, session_finite
    from gateway.session_authorities import owner_scope
    from gateway.session_authority import SessionAuthority
    from gateway.session_hosted_service import ensure_hosted_service
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions

    runner, homes, _, call = mux
    executed = []
    async def execute(authority, ref, row):
        executed.append((authority.profile_id, ref.profile_id, str(get_hermes_home()), row['admission_id']))
        return 'member reply'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    for authority in runner.session_authorities:
        monkeypatch.setattr(authority, '_schedule', SessionAuthority._schedule.__get__(authority))
    call(ensure_hosted_service(runner))
    source = runner.session_authorities.require(homes['alpha'])
    target = runner.session_authorities.require(homes['beta'])
    with owner_scope(source):
        service = source.hosted_room_service
        service.authorize_room('alice', 'room', create=True)
        service.create_room(room_id='room', name='Room', members=[
            {'member_id': 'host', 'profile': 'alpha', 'handle': 'host'},
            {'member_id': 'helper', 'profile': 'beta', 'handle': 'helper'}])
        service.send(room_id='room', event_id='input',
                     payload={'text': '@helper hello', 'thread_id': 'thread'})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        events = hosted_rooms.read_events(source.db.db_path, room_id='room')['events']
        if any(event['kind'] == 'room.activity' and event['payload'].get('status') == 'settled'
               for event in events):
            break
        time.sleep(0.02)
    else:
        pytest.fail(f'no canonical member publication: {service.status("room")}')
    assert len(executed) == 1
    assert executed[0][:3] == (str(homes['beta']),) * 3
    sid, = target.sessions
    row, = list_session_admissions(target.db, session_id=sid, pending_only=False)
    assert row['admission_id'] == executed[0][3]
    assert row['status'] == 'terminal'
    assert not source.sessions and not runner.session_authority.sessions
    assert sum(event['kind'] == 'message.member' for event in events) == 1


@pytest.mark.parametrize('mux', ['starting'], indirect=True)
def test_startup_preserves_queued_work_until_ready(mux, monkeypatch):
    from types import SimpleNamespace
    from gateway import hosted_room_driver as tasks, session_finite
    from gateway.run_runtime import publish_gateway_runtime_ready
    from gateway.session_authorities import owner_scope
    from gateway.session_authority import SessionAuthority
    from gateway.session_hosted_service import CanonicalHostedRoomService, ensure_hosted_service
    from hermes_state_runtime import list_session_admissions

    runner, homes, loop, call = mux
    source = runner.session_authorities.require(homes['alpha'])
    target = runner.session_authorities.require(homes['beta'])
    executed = []
    async def execute(authority, ref, row):
        executed.append(row['admission_id'])
        return 'member reply'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    for authority in runner.session_authorities:
        monkeypatch.setattr(authority, '_schedule', SessionAuthority._schedule.__get__(authority))
    with owner_scope(source):
        service = CanonicalHostedRoomService(source, loop)
        source.hosted_room_service = service
        service.authorize_room('alice', 'startup-room', create=True)
        service.create_room(room_id='startup-room', name='Startup', members=[
            {'member_id': 'host', 'profile': 'alpha', 'handle': 'host'},
            {'member_id': 'helper', 'profile': 'beta', 'handle': 'helper'}])
        service.send(room_id='startup-room', event_id='input',
                     payload={'text': '@helper hello', 'thread_id': 'thread'})
    queued, = tasks.list_tasks(source.db.db_path, room_id='startup-room')
    assert queued['status'] == 'queued'
    call(ensure_hosted_service(runner))
    # If startup released a worker, let its real pre-submit path finish. No
    # exception is injected: the live private descriptor still says starting.
    deadline = time.monotonic() + 3
    while service.runtime.status()['running'] and time.monotonic() < deadline:
        current = tasks.get_task(source.db.db_path, queued['identity'])
        if current['status'] in tasks.TERMINAL_STATUSES:
            break
        time.sleep(0.02)
    current = tasks.get_task(source.db.db_path, queued['identity'])
    assert current['status'] == 'queued', current
    assert current['execution_generation'] == 0
    assert not executed and not target.sessions
    assert all(getattr(a, 'hosted_room_service', None) is not None for a in runner.session_authorities)
    assert all(not a.hosted_room_service.runtime.status()['running'] for a in runner.session_authorities)

    async def ready():
        runner._running = True
        # Only listener-liveness metadata is needed; the private socket is real.
        runner.session_api = SimpleNamespace(task=loop.create_future())
        publish_gateway_runtime_ready(runner)
    call(ready())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = tasks.get_task(source.db.db_path, queued['identity'])
        if current['status'] == 'settled':
            break
        time.sleep(0.02)
    assert current['status'] == 'settled', current
    assert len(executed) == 1
    sid, = target.sessions
    admission, = list_session_admissions(target.db, session_id=sid, pending_only=False)
    assert admission['admission_id'] == executed[0] and admission['status'] == 'terminal'
