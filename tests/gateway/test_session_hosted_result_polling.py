"""A live hosted turn can recover its exact bounded receipt without reading the transcript."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as state, hosted_rooms
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state_runtime import claim_session_input, settle_session_input
from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime


def _room_turn(hosted_owner, monkeypatch, execute):
    """One room turn run by the authority's real FIFO drain, whose completion never reaches the
    room: the membership check refuses that delivery, so the callback is dropped as it is live."""
    from gateway import session_finite
    from gateway.session_authority import SessionAuthority

    authority, loop, principal, agent = hosted_owner
    db, now = authority.db.db_path, [100.0]
    hosted_rooms.create_room(db, room_id='room', name='Batch', authority_gateway_id='gateway-a',
        members=[{'member_id': 'one', 'profile': 'default'}, {'member_id': 'two', 'profile': 'other'}])
    identity = state.TaskIdentity('room', 'task', 'thread', 'turn')
    state.admit_task(db, identity, payload={'target_profile': 'default', 'target_member_id': 'one',
        'source_event_seq': 1, 'prompt': 'input'}, clock=lambda: now[0])
    authority.runner._adapter_for_source = lambda source: authority.runner.adapters[source.platform]
    authority.hosted_room_service = SimpleNamespace(check_admission=lambda ref, row: True)
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    monkeypatch.setattr(authority, '_schedule', SessionAuthority._schedule.__get__(authority))
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='one', profile='default',
        principal=principal, authorize=lambda operation, *_: operation != 'terminal')
    turn = SimpleNamespace(db=db, identity=identity, rpc=rpc, agent=agent, loop=loop, published=[],
                           results=[], history=[], polls=[], on_poll=lambda: None)
    results = rpc.results

    def read_results(**kwargs):
        turn.results.append(kwargs['admission_ids'])
        return results(**kwargs)

    def unreadable_history(**kwargs):
        turn.history.append(kwargs)
        raise TimeoutError('the whole transcript is too slow to read')

    monkeypatch.setattr(rpc, 'results', read_results)
    monkeypatch.setattr(rpc, 'history', unreadable_history)
    turn.binding = HostedRoomBinding('room', 'gateway-a', 1)
    turn.runtime = HostedRoomRuntime(db_path=db, rooms=[turn.binding], rpc=rpc,
        turn_lock=lambda _: nullcontext(), clock=lambda: now[0], active_poll_interval_seconds=0.01,
        publish_terminal=lambda _binding, task: turn.published.append(task))
    wait = turn.runtime._wake.wait

    def poll(timeout):
        turn.polls.append(timeout)
        assert len(turn.polls) < 1000, 'observation never found the outcome'
        turn.on_poll()
        return wait(timeout)

    monkeypatch.setattr(turn.runtime._wake, 'wait', poll)
    return turn


@pytest.mark.parametrize('outcome', ['completed', 'failed'])
def test_observation_finds_the_exact_admission_result_without_history(hosted_owner, monkeypatch, outcome):
    async def execute(_authority, _ref, _row):
        if outcome == 'failed':
            raise RuntimeError('provider unavailable')
        return 'Exact result'

    turn = _room_turn(hosted_owner, monkeypatch, execute)
    turn.runtime._process_room(turn.binding)
    task = state.get_task(turn.db, turn.identity)
    assert task['status'] == ('settled' if outcome == 'completed' else 'failed')
    assert task['result']['text'] == ('Exact result' if outcome == 'completed' else 'The admitted turn failed.')
    admission_ids = {tuple(ids) for ids in turn.results}
    assert len(admission_ids) == 1 and not turn.history
    assert len(turn.published) == 1

    from gateway import session_results
    saved_reads = []
    saved_result = session_results.admission_result

    def saved(*args, **kwargs):
        saved_reads.append(True)
        return saved_result(*args, **kwargs)

    monkeypatch.setattr(session_results, 'admission_result', saved)
    for _ in range(2):
        info = turn.rpc.info(profile='default', session_id=turn.rpc.ref.session_id, source='bot_room')
        assert info['active'] is False
    assert not saved_reads  # Delivered historical replies are not reloaded by every status poll.


def test_stop_of_a_watched_turn_reads_its_receipt_not_the_transcript(hosted_owner, monkeypatch):
    stopped = asyncio.Event()

    async def execute(_authority, _ref, _row):
        await stopped.wait()
        return 'Partial answer before the Stop.'

    turn = _room_turn(hosted_owner, monkeypatch, execute)
    turn.agent.interrupt = lambda: turn.loop.call_soon_threadsafe(stopped.set)

    def stop_once_running():
        if state.get_task(turn.db, turn.identity)['status'] == 'running':
            state.begin_task_cancel(turn.db, turn.identity, cancel_id='room-stop',
                                    expected_cancel_generation=0, clock=turn.runtime.clock)

    turn.on_poll = stop_once_running
    turn.runtime._process_room(turn.binding)
    assert state.get_task(turn.db, turn.identity)['status'] == 'cancelled'
    assert turn.results and not turn.history


@pytest.fixture
def deferred_turn(hosted_owner, tmp_path, request):
    authority, loop, _, agent = hosted_owner
    authority.profile_id = str(tmp_path)
    db, now = authority.db.db_path, [100.0]
    service = CanonicalHostedRoomService(authority, loop)
    service.profile_homes = lambda: {'default': tmp_path, 'other': tmp_path}
    service.runtime.clock = lambda: now[0]
    service.runtime._wake.wait = lambda _: service.runtime._stop.set()
    service.authorize_room('alice', 'room', create=True)
    hosted_rooms.create_room(db, room_id='room', name='Room',
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(), members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
    if getattr(request, 'param', False):
        from gateway.hosted_room_context import set_room_context
        set_room_context(db, room_id='room', event_id='context', purpose='Work', notes='',
                         routing='lead', lead_member_id='one')
    service.send(room_id='room', event_id='input', payload={'text': 'work' if getattr(request, 'param', False) else '@one work', 'thread_id': 'thread'})
    binding, = service.bindings()
    service.runtime._run_room_once(binding)
    task, = state.list_tasks(db, room_id='room', status='running')
    rpc, = service.member_rpcs.values()
    coords = dict(profile='default', source='bot_room', session_id=rpc.ref.session_id)
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=rpc.ref.session_id)
    rpc.callbacks.clear()  # a restarted observer has no completion callbacks
    lease = service.runtime._ensure_lease(binding)
    attempt = state.TaskAttempt(task['identity'], lease, 1, 0)
    state.mark_attempt_indeterminate(db, attempt, lease, clock=service.runtime.clock)
    state.defer_indeterminate_task(db, task['identity'], lease, expected_execution_generation=1,
        expected_cancel_generation=0, reason='member_unavailable', clock=service.runtime.clock)
    return SimpleNamespace(db=db, authority=authority, loop=loop, agent=agent, now=now, service=service,
        runtime=service.runtime, rpc=rpc, coords=coords, row=row, binding=binding, identity=task['identity'],
        control=dict(member_id='one', task_id=task['identity'].task_id, execution_generation=1))


def _finish(room, row, text):
    settle_session_input(room.authority.db, epoch=room.authority.epoch,
        admission_id=row['admission_id'], generation=row['generation'], outcome='completed',
        result={'result': {'final_response': text}, 'usage': {}})


@pytest.mark.parametrize('text', ['x' * 65536, '漢' * 21845], ids=['ascii', 'unicode'])
def test_retry_finds_its_reply_beyond_the_recent_receipt_budget(deferred_turn, text):
    room = deferred_turn
    _finish(room, room.row, 'Already finished once.')
    rpc = HostedRoomAuthorityRPC(room.authority, room.loop, room_id='room', member_id='one',
        profile='default', principal=room.rpc.principal, authorize=lambda *_: True)
    for index in range(8):
        rpc.submit(**room.coords, prompt='later work', task=state.TaskIdentity('room', f'later-{index}', 't', 'u'),
            execution_generation=1, on_terminal=lambda _: None)
        row = claim_session_input(room.authority.db, epoch=room.authority.epoch,
                                  session_id=room.coords['session_id'])
        _finish(room, row, text)
    assert room.identity.task_id not in {r['task_id'] for r in rpc.results(**room.coords)}
    result = room.service.retry_room_task('room', **room.control)
    assert (result['status'], (result['result'] or {}).get('text')) == ('settled', 'Already finished once.')
    assert result['execution_generation'] == 1


@pytest.mark.parametrize('deferred_turn', [False, True], indirect=True, ids=['member', 'lead'])
def test_deferred_receipt_is_observed_when_reads_return(deferred_turn, monkeypatch):
    room = deferred_turn
    _finish(room, room.row, 'The stored answer survived.')
    with monkeypatch.context() as patch:
        patch.setattr(room.rpc, 'results', lambda **_: (_ for _ in ()).throw(TimeoutError()))
        room.now[0] += 301
        room.runtime._stop.clear()
        room.runtime._run_room_once(room.binding)
        assert state.get_task(room.db, room.identity)['status'] == 'deferred'
    room.runtime._stop.clear()
    room.runtime._run_room_once(room.binding)
    task = state.get_task(room.db, room.identity)
    assert (task['status'], (task['result'] or {}).get('text')) == ('settled', 'The stored answer survived.')


def test_retry_waits_for_the_exact_receipt_store_to_answer(deferred_turn, monkeypatch):
    room = deferred_turn
    _finish(room, room.row, 'Only run once.')

    def unavailable(**_kwargs):
        raise TimeoutError('receipt storage unavailable')

    with monkeypatch.context() as patch:
        patch.setattr(room.rpc, 'results', unavailable)
        with pytest.raises(TimeoutError):
            room.service.retry_room_task('room', **room.control)
        assert state.get_task(room.db, room.identity)['status'] == 'deferred'
    assert room.service.retry_room_task('room', **room.control)['result']['text'] == 'Only run once.'
