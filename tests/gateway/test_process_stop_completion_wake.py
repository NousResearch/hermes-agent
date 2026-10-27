"""A finished native job wakes its existing human FIFO, never a producer replay."""
import asyncio
import shlex
import sys
import threading
from types import SimpleNamespace

import pytest
import pytest_asyncio

from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit
from tests.waiting import async_wait_until

pytestmark = pytest.mark.platforms('linux')


def _pause_original_reader(registry, monkeypatch, entered, release):
    ingest = registry._ingest_output
    def paused_ingest(session, text, **kwargs):
        if 'original pre-Stop output' in text:
            entered.set()
            assert release.wait(20)
        ingest(session, text, **kwargs)
    monkeypatch.setattr(registry, '_ingest_output', paused_ingest)


@pytest_asyncio.fixture
async def stopped_job(tmp_path, monkeypatch):
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import process_registry as module
    from tools.process_registry_job import job_quiescent
    db, authority = _authority(tmp_path, monkeypatch)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    entered, release = threading.Event(), threading.Event()
    _pause_original_reader(registry, monkeypatch, entered, release)
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        own = registry.spawn_local(shlex.join([sys.executable, '-c',
            'import threading; print("original pre-Stop output",flush=True); threading.Event().wait(20)']),
            cwd=str(tmp_path), task_id='worker', owner_task_id='worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    state = SimpleNamespace(db=db, authority=authority, registry=registry, own=own,
                            entered=entered, release=release, executed=[])
    async def execute(owner, ref, row):
        assert owner is state.authority
        assert state.own._completion_event.is_set() and job_quiescent(state.own)
        assert state.own._result_persisted_generation == state.own.ownership_generation
        state.executed.append(row['request_id'])
        return 'human continued'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        assert process_stop_pending(db, 's') and not own._completion_event.is_set()
        await _submit(authority, 'only-human')
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert state.executed == [] and process_stop_pending(db, 's')
        yield state
    finally:
        release.set()
        await asyncio.to_thread(own._completion_event.wait, 5)
        await asyncio.to_thread(registry.kill_process, own.id)
        db.close()


async def _finish_original_reader(state):
    from tools.process_registry_job import job_quiescent
    state.release.set()
    assert await asyncio.to_thread(state.own._completion_event.wait, 5)
    await asyncio.to_thread(state.own._reader_thread.join, 5)
    assert job_quiescent(state.own)
    # Wait behind the reader's posted callbacks without assuming a quiet runner.
    barrier = asyncio.Event()
    asyncio.get_running_loop().call_soon(barrier.set)
    await barrier.wait()


@pytest.mark.asyncio
async def test_actual_original_reader_wakes_pending_human_without_another_message(stopped_job):
    from gateway.session_ancillary import process_stop_pending
    state = stopped_job
    await _finish_original_reader(state)
    await async_wait_until(lambda: state.executed == ['only-human'], timeout=5)
    assert process_stop_pending(state.db, 's') is None
    assert state.own.output_buffer.count('original pre-Stop output') == 1
    assert all(value[2] is None for value in state.own._completion_observers.values())


@pytest.mark.asyncio
@pytest.mark.parametrize('invalidated', ['epoch', 'token', 'parent', 'result'])
async def test_completion_cannot_wake_after_native_scope_or_result_changes(stopped_job, monkeypatch, invalidated):
    from gateway.session_ancillary import _PROCESS_STOP_PREFIX, process_stop_pending
    from hermes_state_runtime import begin_runtime_epoch
    from tools import process_registry_results as results
    state, scheduled = stopped_job, []
    monkeypatch.setattr(state.authority, '_schedule', lambda ref: scheduled.append(ref))
    if invalidated == 'epoch':
        begin_runtime_epoch(state.db, instance_id='new-native-owner')
    if invalidated == 'token':
        # An unrecognized/replaced durable Stop marker is uncertainty, not authority.
        state.db.set_meta(_PROCESS_STOP_PREFIX + 's', 'replaced-native-stop')
    if invalidated == 'parent':
        assert state.registry.transfer_ownership(state.own.id, from_owner='worker', to_owner='parent',
            to_task_id='parent', to_session_key='route', to_parent_session_id='another-native-owner',
            note='continue', child_stop_event=None) is state.own
    if invalidated == 'result':
        def refuse_result(*args, **kwargs):
            raise PermissionError('finite original-result refusal')
        monkeypatch.setattr(results, 'atomic_json_write', refuse_result)
    await _finish_original_reader(state)
    assert scheduled == [] and state.executed == []
    assert process_stop_pending(state.db, 's') is not None
    if invalidated == 'result':
        assert state.own._result_persisted_generation == -1
    else:
        assert all(value[2] is None for value in state.own._completion_observers.values())


@pytest.mark.asyncio
async def test_recovered_native_owner_reconstructs_the_completion_wake(stopped_job, monkeypatch, tmp_path):
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_authority import LiveSession, SessionAuthority
    from hermes_state_runtime import begin_runtime_epoch, recover_session_inputs
    from tools import process_registry as module
    state = stopped_job
    old = state.authority
    epoch = begin_runtime_epoch(state.db, instance_id='recovered-native-owner')
    recover_session_inputs(state.db, epoch=epoch)
    authority = SessionAuthority(old.runner, profile_id='owned', instance_id='recovered-native-owner', db=state.db, epoch=epoch)
    authority.sessions['s'] = LiveSession(old.sessions['s'].source, 'route')
    state.authority = authority
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    _pause_original_reader(registry, monkeypatch, threading.Event(), state.release)
    assert registry.recover_from_checkpoint() == 1
    state.own = registry.get(state.own.id)
    authority._schedule(REF)  # Existing recovery pump, without adding another admission.
    await asyncio.wait_for(authority.sessions['s'].task, 10)
    assert state.executed == [] and process_stop_pending(state.db, 's')
    await _finish_original_reader(state)
    await async_wait_until(lambda: state.executed == ['only-human'], timeout=5)
    assert process_stop_pending(state.db, 's') is None


@pytest.mark.asyncio
async def test_finish_between_failed_stop_and_subscription_is_not_a_lost_wake(stopped_job, monkeypatch):
    from gateway.session_ancillary import _observe_process_stop_completion
    state = stopped_job
    # The subscriber identity is process-local only. Model a new owner-side
    # observation after the original publisher has already set completion.
    with state.own._lock:
        state.own._completion_observers.clear()
    await _finish_original_reader(state)
    assert state.executed == []
    _observe_process_stop_completion(state.authority, REF)
    await async_wait_until(lambda: state.executed == ['only-human'], timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize('reload_registry', [False, True], ids=['live-ready-result', 'cold-ready-result'])
async def test_already_ready_result_subscribed_after_cleanup_rechecks_logical_stop(stopped_job, monkeypatch,
                                                                                 reload_registry):
    from gateway.session_ancillary import (_observe_process_stop_completion, _recover_stopped_processes,
                                           process_stop_pending)
    from tools import process_registry as module
    from tools.process_registry_results import save_completed_result
    state = stopped_job
    with state.own._lock:
        state.own._completion_observers.clear()
    await _finish_original_reader(state)
    assert save_completed_result(state.own)
    assert not state.own._scope_stop_pending and state.executed == []
    assert process_stop_pending(state.db, 's')
    if reload_registry:
        registry = module.ProcessRegistry()
        monkeypatch.setattr(module, 'process_registry', registry)
        assert _recover_stopped_processes(state.authority, REF)
        state.own = registry.get(state.own.id)
        assert await asyncio.to_thread(state.own._completion_event.wait, 5)
    assert state.own._completion_event.is_set() and state.own._result_complete
    assert state.own._result_persisted_generation == state.own.ownership_generation
    assert not state.own._completion_observers
    _observe_process_stop_completion(state.authority, REF)
    _observe_process_stop_completion(state.authority, REF)
    await async_wait_until(lambda: state.executed == ['only-human'], timeout=5)
    assert process_stop_pending(state.db, 's') is None
    assert all(value[2] is None for value in state.own._completion_observers.values())


@pytest.mark.asyncio
async def test_ready_observation_dedup_does_not_poll_an_independent_unknown(stopped_job, monkeypatch):
    from gateway import session_ancillary
    from tools.process_registry_results import save_completed_result
    state = stopped_job
    with state.own._lock:
        state.own._completion_observers.clear()
    await _finish_original_reader(state)
    assert save_completed_result(state.own) and not state.own._scope_stop_pending
    attempts = []
    def unavailable_owner(*args):
        attempts.append(REF)
        return False  # Component fault: actual completed job cannot waive another uncertainty.
    monkeypatch.setattr(session_ancillary, '_recover_stopped_processes', unavailable_owner)
    session_ancillary._observe_process_stop_completion(state.authority, REF)
    session_ancillary._observe_process_stop_completion(state.authority, REF)
    await async_wait_until(lambda: bool(attempts) and state.authority.sessions['s'].task.done(), timeout=5)
    assert attempts == [REF] and state.executed == []
    assert session_ancillary.process_stop_pending(state.db, 's')
    session_ancillary._observe_process_stop_completion(state.authority, REF)
    barrier = asyncio.Event()
    asyncio.get_running_loop().call_soon(barrier.set)
    await barrier.wait()
    assert attempts == [REF] and state.executed == []


@pytest.mark.asyncio
async def test_completion_wake_keeps_its_profile_when_the_loop_serves_another(stopped_job, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    state = stopped_job
    other = tmp_path / 'another-profile'
    token = set_hermes_home_override(other)
    try:
        # Reader and callback retain their captured owner context; the loop's
        # current request may meanwhile belong to a different profile.
        state.release.set()
        await async_wait_until(lambda: state.executed == ['only-human'], timeout=5)
        assert not (other / 'state.db').exists()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_result_success_posts_wake_while_caller_holds_registry_lock(stopped_job, monkeypatch):
    import signal
    from tools import process_registry_results as results
    state = stopped_job
    write = results.atomic_json_write
    def refused(*args, **kwargs):
        raise PermissionError('finite final-result refusal')
    monkeypatch.setattr(results, 'atomic_json_write', refused)
    await _finish_original_reader(state)
    assert state.executed == [] and state.own._result_persisted_generation == -1
    monkeypatch.setattr(results, 'atomic_json_write', write)
    original = signal.getsignal(signal.SIGALRM)
    def lock_timeout(*args):
        raise RuntimeError('result success reacquired its caller registry lock')
    signal.signal(signal.SIGALRM, lock_timeout)
    signal.alarm(5)
    try:
        with state.registry._lock:  # Real non-reentrant native lock, not a fake RLock.
            assert results.save_completed_result(state.own)
            assert state.executed == []  # The concrete observer only posts to the loop.
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original)
    await async_wait_until(lambda: state.executed == ['only-human'], timeout=5)


@pytest.mark.asyncio
async def test_old_completion_cannot_wake_a_genuinely_newer_native_stop(stopped_job, monkeypatch, tmp_path):
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_context import clear_session_vars, set_session_vars
    state = stopped_job
    deferred = []
    with state.own._lock:
        for key, (observation, expected, completed) in state.own._completion_observers.items():
            state.own._completion_observers[key] = (observation, expected, lambda completed=completed: deferred.append(completed))
    await _finish_original_reader(state)
    assert deferred and state.executed == []
    entered, finish = asyncio.Event(), asyncio.Event()
    async def executing_human(owner, ref, row):
        entered.set()
        await finish.wait()
        return 'original human settled'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', executing_human)
    monkeypatch.setattr(state.authority.runner, '_cached_agent_for', lambda _route: None, raising=False)
    await state.authority.interrupt(ACTOR, REF, state.authority._handle(REF).execution_generation)
    await asyncio.wait_for(entered.wait(), 5)
    assert state.authority._handle(REF).execution_generation == 1
    second_entered, second_release = threading.Event(), threading.Event()
    _pause_original_reader(state.registry, monkeypatch, second_entered, second_release)
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        second = state.registry.spawn_local(shlex.join([sys.executable, '-c',
            'import threading; print("original pre-Stop output",flush=True); threading.Event().wait(20)']),
            cwd=str(tmp_path), task_id='second-worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    try:
        assert await asyncio.to_thread(second_entered.wait, 5)
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await state.authority.interrupt(ACTOR, REF, 1)
        token = process_stop_pending(state.db, 's')
        assert token.endswith(':1')
        scheduled = []
        schedule = state.authority._schedule
        monkeypatch.setattr(state.authority, '_schedule', lambda ref: scheduled.append(ref))
        for completed in deferred:
            completed()
        barrier = asyncio.Event()
        asyncio.get_running_loop().call_soon(barrier.set)
        await barrier.wait()
        assert scheduled == [] and process_stop_pending(state.db, 's') == token
        monkeypatch.setattr(state.authority, '_schedule', schedule)
    finally:
        second_release.set()
        await asyncio.to_thread(second._completion_event.wait, 5)
        finish.set()
        await asyncio.wait_for(state.authority.sessions['s'].task, 10)


@pytest.mark.asyncio
async def test_native_retry_keeps_the_original_complete_reader_instead_of_replacing_it(stopped_job, monkeypatch):
    state = stopped_job
    cold_started, release_cold = threading.Event(), threading.Event()
    cold_readers = []
    reader = state.registry._log_reader_loop
    def held_cold_reader(session, *args, **kwargs):
        cold_readers.append(session)
        cold_started.set()
        assert release_cold.wait(10)
        return reader(session, *args, **kwargs)
    monkeypatch.setattr(state.registry, '_log_reader_loop', held_cold_reader)
    try:
        await _finish_original_reader(state)
        await async_wait_until(lambda: bool(state.executed) or cold_started.is_set(), timeout=5)
        assert state.executed == ['only-human'], 'Recovery discarded the original complete durable reader'
        assert not cold_started.is_set()
        assert state.registry.get(state.own.id) is state.own
    finally:
        release_cold.set()
        for process in cold_readers:
            await asyncio.to_thread(process._completion_event.wait, 5)
