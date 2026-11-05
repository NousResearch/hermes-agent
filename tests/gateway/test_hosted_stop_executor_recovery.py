"""Old stopping attempts recover through real cross-profile control sockets."""
from concurrent.futures import ThreadPoolExecutor
import time

import pytest

pytest_plugins = ('tests.gateway.test_hosted_mux_runtime',)


def _seed_old_stopping_cross_profile_task(mux, room_id='room'):
    from gateway import hosted_room_driver as tasks, hosted_rooms
    from gateway.session_authorities import owner_scope
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway.session_hosted_transport import install_hosted_transport
    from tui_gateway.hosted_room_driver import HostedRoomBinding

    runner, homes, loop, _ = mux
    for authority in runner.session_authorities:
        with owner_scope(authority):
            if getattr(authority, 'hosted_room_service', None) is None:
                service = CanonicalHostedRoomService(authority, loop)
                authority.hosted_room_service = service
                install_hosted_transport(runner.session_control_server, authority, loop, attest=service.attest)
    source = runner.session_authorities.require(homes['alpha'])
    service = source.hosted_room_service
    with owner_scope(source):
        service.authorize_room('alice', room_id, create=True)
        hosted_rooms.create_room(source.db.db_path, room_id=room_id, name='Room',
            authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
            members=[{'member_id': 'helper', 'profile': 'beta', 'handle': 'helper'}])
    identity = tasks.TaskIdentity(room_id, 'old-task', 'thread', 'turn')
    tasks.admit_task(source.db.db_path, identity, payload={
        'target_profile': 'beta', 'target_member_id': 'helper', 'source_event_seq': 1,
        'prompt': 'input'}, clock=time.time)
    binding = HostedRoomBinding(room_id, hosted_rooms.local_authority_gateway_id(), 1)
    old = tasks.acquire_lease(source.db.db_path, room_id=room_id, gateway_id=binding.gateway_id,
        authority_epoch=1, process_generation='previous-process', ttl_seconds=1,
        clock=lambda: time.time() - 60)
    tasks.start_task(source.db.db_path, identity, old, expected_cancel_generation=0,
        clock=lambda: time.time() - 60)
    task = tasks.get_task(source.db.db_path, identity)
    rpc = service._resolve_member_transport(binding, task)
    sid = rpc.create(profile='beta', source='bot_room', title=f'Group: {room_id}')['session_id']
    rpc.submit(profile='beta', source='bot_room', session_id=sid, prompt='input', task=identity,
        execution_generation=1, on_terminal=None)
    tasks.begin_task_cancel(source.db.db_path, identity, cancel_id='old-stop',
        expected_cancel_generation=0, clock=time.time)
    service.runtime.publish_terminal = None
    return service, binding, identity, rpc, sid


@pytest.mark.parametrize('admission_state', ['queued', 'unknown'])
def test_cross_profile_stopping_recovery(mux, admission_state):
    from gateway import hosted_room_driver as tasks
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input, recover_session_inputs

    service, binding, identity, rpc, sid = _seed_old_stopping_cross_profile_task(mux)
    runner, homes, _, _ = mux
    if admission_state == 'unknown':
        target = runner.session_authorities.require(homes['beta'])
        claim_session_input(target.db, epoch=target.epoch, session_id=sid)
        target.epoch = begin_runtime_epoch(target.db, instance_id='restarted')
        assert recover_session_inputs(target.db, epoch=target.epoch) == 1
        target.sessions.pop(sid)
    lease = service.runtime._ensure_lease(binding)
    assert service.runtime._retry_stopping_tasks(binding, lease) is False
    assert tasks.get_task(service.db_path, identity)['status'] == 'cancelled'
    assert not rpc.callbacks


@pytest.mark.platforms('linux')
def test_cross_profile_stopping_recovery_with_one_default_worker(mux, monkeypatch):
    from gateway import hosted_room_driver as tasks
    from gateway import session_hosted_transport as transport

    service, binding, identity, rpc, _sid = _seed_old_stopping_cross_profile_task(mux)
    _, _, loop, call = mux
    lease = service.runtime._ensure_lease(binding)
    saved = tasks.get_task(service.db_path, identity)
    assert saved['run_process_generation'] != lease.process_generation
    original = transport.owner_request

    def bounded(*args, **kwargs):
        kwargs['timeout'] = 5
        return original(*args, **kwargs)

    monkeypatch.setattr(transport, 'owner_request', bounded)

    async def narrow_pool():
        previous = loop._default_executor
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        return previous

    previous = call(narrow_pool())
    try:
        assert service.runtime._retry_stopping_tasks(binding, lease) is False
        assert tasks.get_task(service.db_path, identity)['status'] == 'cancelled'
    finally:
        narrow = loop._default_executor
        loop.set_default_executor(previous)
        narrow.shutdown(wait=True)

    assert not rpc.callbacks


@pytest.mark.platforms('linux')
def test_two_rooms_recover_stops_concurrently(mux):
    from gateway import hosted_room_driver as tasks

    seeded = [_seed_old_stopping_cross_profile_task(mux, room_id) for room_id in ('first', 'second')]

    def recover(seed):
        service, binding, identity, rpc, _sid = seed
        lease = service.runtime._ensure_lease(binding)
        assert service.runtime._retry_stopping_tasks(binding, lease) is False
        assert not rpc.callbacks
        return tasks.get_task(service.db_path, identity)['status']

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(recover, seeded)) == ['cancelled', 'cancelled']


@pytest.mark.platforms('linux')
def test_unavailable_member_leaves_stop_pending_until_control_recovers(mux, monkeypatch):
    from gateway import hosted_room_driver as tasks
    from hermes_state_runtime import RuntimeStoreError

    service, binding, identity, rpc, _sid = _seed_old_stopping_cross_profile_task(mux)
    runner, _, _, _ = mux
    lease = service.runtime._ensure_lease(binding)

    def unavailable(params, peer):
        raise RuntimeStoreError('runtime_draining')

    with monkeypatch.context() as patch:
        patch.setitem(runner.session_control_server.private_handlers, 'hosted-producer', unavailable)
        assert service.runtime._retry_stopping_tasks(binding, lease) is True
        assert tasks.get_task(service.db_path, identity)['status'] == 'stopping'
    assert service.runtime._retry_stopping_tasks(binding, lease) is False
    assert tasks.get_task(service.db_path, identity)['status'] == 'cancelled'
    assert not rpc.callbacks
