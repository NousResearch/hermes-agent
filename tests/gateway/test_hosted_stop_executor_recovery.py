"""Old stopping attempts recover through real cross-profile control sockets."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from gateway.session_authority import admission_row
from gateway.session_persistence import authority_persistence
from persistence.core.errors import AuthorityRefusal
from persistence.gateway.hosted_rooms import HostedTaskIdentity
from tests.fixtures.storage_seed import seed_messages

pytest_plugins = ('tests.gateway.test_hosted_mux_runtime',)


def _admissions(authority, sid, pending_only=True):
    return [admission_row(record) for record in authority_persistence(authority).admissions.list(
        session_id=sid, pending_only=pending_only)]


def _claim(authority, sid):
    return admission_row(authority_persistence(authority).admissions.claim(epoch=authority.epoch, session_id=sid))


def _restart_target(target, sid):
    """The target's input claimed, then a restart: a new epoch whose recovery marks it ``unknown``."""
    _claim(target, sid)
    store = authority_persistence(target)
    target.epoch = store.runtime.begin_epoch(instance_id='restarted')
    return store.admissions.recover(epoch=target.epoch)


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
        service.insert_room(room_id=room_id, name='Room', owner='alice',
            members=[{'member_id': 'host', 'profile': 'alpha', 'handle': 'host'},
                     {'member_id': 'helper', 'profile': 'beta', 'handle': 'helper'}])
        service.send(room_id=room_id, event_id='initial-input',
                     payload={'text': '@helper initial input', 'thread_id': 'thread'})
    store, placement = service.store, service.placement
    queued, = tasks.list_tasks(store, placement=placement, room_id=room_id)
    identity = queued['identity']
    binding = HostedRoomBinding(room_id, hosted_rooms.local_authority_gateway_id(), 1)
    old = tasks.acquire_lease(store, placement=placement, room_id=room_id, gateway_id=binding.gateway_id,
        authority_epoch=1, process_generation='previous-process', ttl_seconds=1,
        clock=lambda: time.time() - 60)
    tasks.start_task(store, identity, old, placement=placement, expected_cancel_generation=0,
        clock=lambda: time.time() - 60)
    task = tasks.get_task(store, identity, placement=placement)
    rpc = service._resolve_member_transport(binding, task)
    sid = rpc.create(profile='beta', source='bot_room', title=f'Group: {room_id}')['session_id']
    rpc.submit(profile='beta', source='bot_room', session_id=sid, prompt=task['payload']['prompt'], task=identity,
        execution_generation=1, on_terminal=None)
    hosted_rooms.request_room_stop(store, placement=placement, room_id=room_id, cancel_id='old-stop',
        expected_gateway_id=binding.gateway_id, expected_epoch=binding.authority_epoch)
    tasks.begin_task_cancel(store, identity, placement=placement, cancel_id='old-stop',
        expected_cancel_generation=0, clock=time.time)
    service.runtime.publish_terminal = None
    return service, binding, identity, rpc, sid


@pytest.mark.parametrize('admission_state', ['queued', 'unknown'])
def test_cross_profile_stopping_recovery(mux, admission_state):
    from gateway import hosted_room_driver as tasks

    service, binding, identity, rpc, sid = _seed_old_stopping_cross_profile_task(mux)
    runner, homes, _, _ = mux
    target = runner.session_authorities.require(homes['beta'])
    seed_messages(authority_persistence(target), sid, {'role': 'assistant', 'content': 'Preserve this existing group history.'})
    if admission_state == 'unknown':
        assert _restart_target(target, sid) == 1
        target.sessions.pop(sid)
    lease = service.runtime._ensure_lease(binding)
    assert service.runtime._retry_stopping_tasks(binding, lease) is False
    assert tasks.get_task(service.store, identity, placement=service.placement)['status'] == 'cancelled'
    # Clearing the room badge is insufficient: the canonical FIFO must be unblocked too.
    assert all(row['status'] == 'terminal' for row in _admissions(target, sid, pending_only=False))
    assert any(row.content == 'Preserve this existing group history.'
               for row in authority_persistence(target).messages.read_transcript(sid))
    # The next group turn can be admitted to the same member session without replaying the old one.
    following = HostedTaskIdentity(binding.room_id, 'following-task', 'thread', 'following-turn')
    tasks.admit_task(service.store, following, placement=service.placement, payload={
        'target_profile': 'beta', 'target_member_id': 'helper', 'source_event_seq': 2,
        'prompt': 'Continue after the stopped turn.'}, clock=time.time)
    tasks.start_task(service.store, following, lease, placement=service.placement, expected_cancel_generation=0,
                     clock=time.time)
    accepted = rpc.submit(profile='beta', source='bot_room', session_id=sid,
        prompt='Continue after the stopped turn.', task=following, execution_generation=1, on_terminal=None)
    assert accepted['admission_id']
    assert service.runtime.cancel(following, cancel_id='cleanup-following')['status'] == 'cancelled'
    assert not rpc.callbacks


@pytest.mark.platforms('linux')
def test_recovered_group_loads_and_answers_the_next_message_without_replaying_old_work(mux, monkeypatch):
    from gateway import hosted_room_driver as tasks, hosted_rooms, session_finite
    from gateway.session_authority import SessionAuthority
    from tests.waiting import wait_until

    service, binding, identity, rpc, sid = _seed_old_stopping_cross_profile_task(mux)
    runner, homes, _, _ = mux
    target = runner.session_authorities.require(homes['beta'])
    seed_messages(authority_persistence(target), sid, {'role': 'assistant', 'content': 'Earlier conversation is retained.'})
    _restart_target(target, sid)
    target.sessions.pop(sid)
    service.runtime.publish_terminal = service.publish_terminal
    service.runtime._process_room(binding)
    assert tasks.get_task(service.store, identity, placement=service.placement)['status'] == 'cancelled'
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
            events = hosted_rooms.read_events(service.store, placement=service.placement, room_id=binding.room_id)['events']
            return any(e['kind'] == 'message.member' and e['payload'].get('text') ==
                       'The new group message completed.' for e in events)
        wait_until(completed, timeout=15, what='new group reply after recovered Stop')
        wait_until(lambda: not service.status(binding.room_id)['working'], timeout=10)
        assert len(executed) == 1
        assert any(m.content == 'Earlier conversation is retained.'
                   for m in authority_persistence(target).messages.read_transcript(sid))
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
            assert tasks.get_task(service.store, identity, placement=service.placement)['status'] == 'cancelled'
            assert service.status(binding.room_id)['working'] is False
        finally:
            release.set()
            processing.result(timeout=5)


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('mismatch', ['task', 'generation', 'active'])
def test_recovered_stop_does_not_discard_a_different_canonical_attempt(mux, monkeypatch, mismatch):
    from gateway import hosted_room_driver as tasks

    service, binding, identity, rpc, sid = _seed_old_stopping_cross_profile_task(mux)
    runner, homes, _, _ = mux
    target = runner.session_authorities.require(homes['beta'])
    _restart_target(target, sid)
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
    assert tasks.get_task(service.store, identity, placement=service.placement)['status'] == 'stopping'
    assert _admissions(target, sid)[0]['status'] == 'unknown'
    assert service.runtime._retry_stopping_tasks(binding, lease) is False


@pytest.mark.platforms('linux')
def test_cross_profile_stopping_recovery_with_one_default_worker(mux, monkeypatch):
    from gateway import hosted_room_driver as tasks
    from gateway import session_hosted_transport as transport

    service, binding, identity, rpc, _sid = _seed_old_stopping_cross_profile_task(mux)
    _, _, loop, call = mux
    lease = service.runtime._ensure_lease(binding)
    saved = tasks.get_task(service.store, identity, placement=service.placement)
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
        assert tasks.get_task(service.store, identity, placement=service.placement)['status'] == 'cancelled'
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
        return tasks.get_task(service.store, identity, placement=service.placement)['status']

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(recover, seeded)) == ['cancelled', 'cancelled']


@pytest.mark.platforms('linux')
def test_unavailable_member_leaves_stop_pending_until_control_recovers(mux, monkeypatch):
    from gateway import hosted_room_driver as tasks

    service, binding, identity, rpc, _sid = _seed_old_stopping_cross_profile_task(mux)
    runner, _, _, _ = mux
    lease = service.runtime._ensure_lease(binding)

    def unavailable(params, peer):
        raise AuthorityRefusal('runtime_draining')

    with monkeypatch.context() as patch:
        patch.setitem(runner.session_control_server.private_handlers, 'hosted-producer', unavailable)
        assert service.runtime._retry_stopping_tasks(binding, lease) is True
        assert tasks.get_task(service.store, identity, placement=service.placement)['status'] == 'stopping'
    assert service.runtime._retry_stopping_tasks(binding, lease) is False
    assert tasks.get_task(service.store, identity, placement=service.placement)['status'] == 'cancelled'
    assert not rpc.callbacks


@pytest.mark.platforms('linux')
def test_saved_stop_publishes_a_retained_completed_answer_once_not_cancelled(mux):
    """A lost attempt whose answer this owner still holds (its settlement write failed) is
    resolved as that answer by Stop recovery: the room publishes it once, never a cancel."""
    from gateway import hosted_room_driver as tasks, hosted_rooms
    from gateway.session_settlement_recovery import _mark_unknown

    service, binding, identity, rpc, sid = _seed_old_stopping_cross_profile_task(mux)
    runner, homes, _, _ = mux
    target = runner.session_authorities.require(homes['beta'])
    row = _claim(target, sid)
    target.pending_results[row['admission_id']] = {
        'result': {'final_response': 'The completed answer must survive.', 'completed': True}, 'usage': {}}
    assert _mark_unknown(target, row) == 'unknown'
    service.runtime.publish_terminal = service.publish_terminal

    def published():
        events = hosted_rooms.read_events(service.store, placement=service.placement, room_id=binding.room_id)['events']
        return [e for e in events if e['kind'] == 'message.member'
                and e['payload'].get('text') == 'The completed answer must survive.']

    for _ in range(2):  # recovery runs every cycle: the second pass is a no-op
        service.runtime._process_room(binding)
        hosted = tasks.get_task(service.store, identity, placement=service.placement)
        assert hosted['status'] == 'settled'
        assert len(published()) == 1
    canonical, = _admissions(target, sid, pending_only=False)
    assert (canonical['status'], canonical['outcome']) == ('terminal', 'completed')
    assert service.status(binding.room_id)['working'] is False
    assert not rpc.callbacks


@pytest.mark.platforms('linux')
def test_replayed_discard_returns_the_committed_answer(mux):
    """An exact Discard replay after the first committed a recovered answer gets that answer back
    (an operator Discard that timed out after the commit can still finish)."""
    from gateway.session_settlement_recovery import _mark_unknown

    _, _, identity, rpc, sid = _seed_old_stopping_cross_profile_task(mux)
    runner, homes, _, _ = mux
    target = runner.session_authorities.require(homes['beta'])
    row = _claim(target, sid)
    target.pending_results[row['admission_id']] = {
        'result': {'final_response': 'Recovered once.', 'completed': True}, 'usage': {}}
    assert _mark_unknown(target, row) == 'unknown'
    args = dict(profile='beta', source='bot_room', session_id=sid,
                expected_task_id=identity.task_id, execution_generation=1)
    first, replay = rpc.discard(**args), rpc.discard(**args)
    assert first['status'] == replay['status'] == 'settled'
    assert first['text'] == replay['text'] == 'Recovered once.'


@pytest.mark.platforms('linux')
def test_room_status_answers_while_a_room_checkpoint_is_slow(mux, monkeypatch):
    """Desktop's room status (groups.state) must not wait behind one room's slow preparation:
    the reporter's view timed out after 30 s while a room's checkpoint work held the policy lock."""
    service, binding, _, _, _ = _seed_old_stopping_cross_profile_task(mux, 'slow-room')
    entered, release = threading.Event(), threading.Event()
    original = service.policy_checkpoint.snapshot

    def slow_checkpoint(**kwargs):
        if kwargs['room_id'] == binding.room_id and not entered.is_set():
            entered.set()
            assert release.wait(10)
        return original(**kwargs)

    monkeypatch.setattr(service.policy_checkpoint, 'snapshot', slow_checkpoint)
    with ThreadPoolExecutor(max_workers=2) as pool:
        preparing = pool.submit(service.prepare_room, binding)
        assert entered.wait(5)
        try:
            snapshot = pool.submit(service.status, binding.room_id).result(timeout=3)
            assert snapshot['counts'].get('stopping') == 1
        finally:
            release.set()
            preparing.result(timeout=5)
