"""Old stopping attempts recover through real cross-profile control sockets."""
from concurrent.futures import ThreadPoolExecutor
import threading
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
            members=[{'member_id': 'host', 'profile': 'alpha', 'handle': 'host'},
                     {'member_id': 'helper', 'profile': 'beta', 'handle': 'helper'}])
        service.send(room_id=room_id, event_id='initial-input',
                     payload={'text': '@helper initial input', 'thread_id': 'thread'})
    queued, = tasks.list_tasks(source.db.db_path, room_id=room_id)
    identity = queued['identity']
    binding = HostedRoomBinding(room_id, hosted_rooms.local_authority_gateway_id(), 1)
    old = tasks.acquire_lease(source.db.db_path, room_id=room_id, gateway_id=binding.gateway_id,
        authority_epoch=1, process_generation='previous-process', ttl_seconds=1,
        clock=lambda: time.time() - 60)
    tasks.start_task(source.db.db_path, identity, old, expected_cancel_generation=0,
        clock=lambda: time.time() - 60)
    task = tasks.get_task(source.db.db_path, identity)
    rpc = service._resolve_member_transport(binding, task)
    sid = rpc.create(profile='beta', source='bot_room', title=f'Group: {room_id}')['session_id']
    rpc.submit(profile='beta', source='bot_room', session_id=sid, prompt=task['payload']['prompt'], task=identity,
        execution_generation=1, on_terminal=None)
    hosted_rooms.request_room_stop(service.db_path, room_id=room_id, cancel_id='old-stop',
        expected_gateway_id=binding.gateway_id, expected_epoch=binding.authority_epoch)
    tasks.begin_task_cancel(source.db.db_path, identity, cancel_id='old-stop',
        expected_cancel_generation=0, clock=time.time)
    service.runtime.publish_terminal = None
    return service, binding, identity, rpc, sid


@pytest.mark.parametrize('admission_state', ['queued', 'unknown'])
def test_cross_profile_stopping_recovery(mux, admission_state):
    from gateway import hosted_room_driver as tasks
    from hermes_state_runtime import (
        begin_runtime_epoch, claim_session_input, list_session_admissions, recover_session_inputs,
    )

    service, binding, identity, rpc, sid = _seed_old_stopping_cross_profile_task(mux)
    runner, homes, _, _ = mux
    target = runner.session_authorities.require(homes['beta'])
    target.db.append_message(sid, 'assistant', 'Preserve this existing group history.')
    if admission_state == 'unknown':
        claim_session_input(target.db, epoch=target.epoch, session_id=sid)
        target.epoch = begin_runtime_epoch(target.db, instance_id='restarted')
        assert recover_session_inputs(target.db, epoch=target.epoch) == 1
        target.sessions.pop(sid)
    lease = service.runtime._ensure_lease(binding)
    assert service.runtime._retry_stopping_tasks(binding, lease) is False
    assert tasks.get_task(service.db_path, identity)['status'] == 'cancelled'
    # Clearing the room badge is insufficient: the canonical FIFO must be unblocked too.
    assert all(row['status'] == 'terminal' for row in list_session_admissions(
        target.db, session_id=sid, pending_only=False))
    assert any(row['content'] == 'Preserve this existing group history.'
               for row in target.db.get_messages(sid))
    # The next group turn can be admitted to the same member session without replaying the old one.
    following = tasks.TaskIdentity(binding.room_id, 'following-task', 'thread', 'following-turn')
    tasks.admit_task(service.db_path, following, payload={
        'target_profile': 'beta', 'target_member_id': 'helper', 'source_event_seq': 2,
        'prompt': 'Continue after the stopped turn.'}, clock=time.time)
    tasks.start_task(service.db_path, following, lease, expected_cancel_generation=0, clock=time.time)
    accepted = rpc.submit(profile='beta', source='bot_room', session_id=sid,
        prompt='Continue after the stopped turn.', task=following, execution_generation=1, on_terminal=None)
    assert accepted['admission_id']
    assert service.runtime.cancel(following, cancel_id='cleanup-following')['status'] == 'cancelled'
    assert not rpc.callbacks


@pytest.mark.platforms('linux')
def test_recovered_group_loads_and_answers_the_next_message_without_replaying_old_work(mux, monkeypatch):
    from gateway import hosted_room_driver as tasks, hosted_rooms, session_finite
    from gateway.session_authority import SessionAuthority
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input, recover_session_inputs
    from tests.waiting import wait_until

    service, binding, identity, rpc, sid = _seed_old_stopping_cross_profile_task(mux)
    runner, homes, _, _ = mux
    target = runner.session_authorities.require(homes['beta'])
    target.db.append_message(sid, 'assistant', 'Earlier conversation is retained.')
    claim_session_input(target.db, epoch=target.epoch, session_id=sid)
    target.epoch = begin_runtime_epoch(target.db, instance_id='restarted')
    recover_session_inputs(target.db, epoch=target.epoch)
    target.sessions.pop(sid)
    service.runtime.publish_terminal = service.publish_terminal
    service.runtime._process_room(binding)
    assert tasks.get_task(service.db_path, identity)['status'] == 'cancelled'
    assert service.status(binding.room_id)['working'] is False
    assert not service.status(binding.room_id)['blocked']
    executed = []

    async def execute(authority, ref, row):
        executed.append(row['admission_id'])
        return 'The new group message completed.'

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    for authority in runner.session_authorities:
        monkeypatch.setattr(authority, '_schedule', SessionAuthority._schedule.__get__(authority))
    service.start()
    try:
        service.send(room_id=binding.room_id, event_id='following-input',
                     payload={'text': '@helper continue', 'thread_id': 'following-thread'})
        def completed():
            events = hosted_rooms.read_events(service.db_path, room_id=binding.room_id)['events']
            return any(e['kind'] == 'message.member' and e['payload'].get('text') ==
                       'The new group message completed.' for e in events)
        wait_until(completed, timeout=15, what='new group reply after recovered Stop')
        wait_until(lambda: not service.status(binding.room_id)['working'], timeout=10)
        assert len(executed) == 1
        assert any(m['content'] == 'Earlier conversation is retained.' for m in target.db.get_messages(sid))
    finally:
        assert service.stop(timeout=5)
    assert not rpc.callbacks


@pytest.mark.platforms('linux')
def test_saved_stop_finishes_before_slow_preparation_of_its_own_room(mux, monkeypatch):
    from gateway import hosted_room_driver as tasks

    service, binding, identity, _, _ = _seed_old_stopping_cross_profile_task(mux)
    entered, release = threading.Event(), threading.Event()
    original = service.runtime.prepare_room

    def slow_prepare(room):
        entered.set()
        assert release.wait(10)
        original(room)

    monkeypatch.setattr(service.runtime, 'prepare_room', slow_prepare)
    with ThreadPoolExecutor(max_workers=1) as pool:
        processing = pool.submit(service.runtime._process_room, binding)
        assert entered.wait(5)
        try:
            assert tasks.get_task(service.db_path, identity)['status'] == 'cancelled'
            assert service.status(binding.room_id)['working'] is False
        finally:
            release.set()
            processing.result(timeout=5)


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('mismatch', ['task', 'generation', 'active'])
def test_recovered_stop_does_not_discard_a_different_canonical_attempt(mux, monkeypatch, mismatch):
    from gateway import hosted_room_driver as tasks
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input, list_session_admissions, recover_session_inputs

    service, binding, identity, rpc, sid = _seed_old_stopping_cross_profile_task(mux)
    runner, homes, _, _ = mux
    target = runner.session_authorities.require(homes['beta'])
    claim_session_input(target.db, epoch=target.epoch, session_id=sid)
    target.epoch = begin_runtime_epoch(target.db, instance_id='restarted')
    recover_session_inputs(target.db, epoch=target.epoch)
    target.sessions.pop(sid)
    original = rpc.info

    def other_attempt(**kwargs):
        info = original(**kwargs)
        if mismatch == 'task':
            info['task_id'] = 'another-task'
        elif mismatch == 'generation':
            info['execution_generation'] += 1
        else:
            info['active'] = True
        return info

    def never_discard(**kwargs):
        pytest.fail('Stop must not discard another canonical attempt')

    lease = service.runtime._ensure_lease(binding)
    with monkeypatch.context() as patch:
        patch.setattr(rpc, 'info', other_attempt)
        patch.setattr(rpc, 'discard', never_discard)
        assert service.runtime._retry_stopping_tasks(binding, lease) is True
    assert tasks.get_task(service.db_path, identity)['status'] == 'stopping'
    assert list_session_admissions(target.db, session_id=sid)[0]['status'] == 'unknown'
    assert service.runtime._retry_stopping_tasks(binding, lease) is False


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
