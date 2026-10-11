"""A settlement waiting on a contended SQLite writer never stalls the owner event loop.

The settlement worker holds no stream lock across its write, so loop-side attach/replay/pending
publication keep running; the commit still looks atomic with its completion frame to them.
"""
import asyncio
import sqlite3
import threading
import time

import pytest

from gateway.session_contract import Principal, Submission
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit

_HOLD_S = 1.5
VIEWER = Principal('human', 'owned', frozenset({'session:read', 'session:submit'}), 'viewer')


async def _settling_behind_a_held_writer(tmp_path, monkeypatch):
    """Run a turn whose settlement write waits on a real competing SQLite writer."""
    from gateway import session_results
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    frames, entered, released = [], threading.Event(), threading.Event()
    authority.sessions['s'].event_stream.observers.add(frames.append)
    settle = session_results.settle_session_input

    def contended_settle(*args, **kwargs):
        entered.set()
        return settle(*args, **kwargs)
    monkeypatch.setattr(session_results, 'settle_session_input', contended_settle)

    async def execute(owner, ref, row):
        owner.pending_results[row['admission_id']] = {'result': {'final_response': 'done'}, 'usage': {}}
        # Another process holds the write lock while this turn settles.
        writer = sqlite3.connect(str(tmp_path / 'state.db'), isolation_level=None, check_same_thread=False)
        writer.execute('BEGIN IMMEDIATE')

        def release():
            time.sleep(_HOLD_S)
            released.set()
            writer.rollback()
            writer.close()
        threading.Thread(target=release, daemon=True).start()
        return 'done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    await authority.submit(ACTOR, Submission('work', REF, {'text': 'work'}, 'queue'))
    drain = asyncio.create_task(authority._drain(REF))
    while not entered.is_set():
        await asyncio.sleep(0.01)
    return db, authority, frames, released, drain


def _types(frames):
    return [(f['params']['type'], f['params']['payload'].get('running')) for f in frames]


@pytest.mark.asyncio
async def test_attach_during_a_contended_settlement_leaves_the_loop_free(tmp_path, monkeypatch):
    db, authority, frames, released, drain = await _settling_behind_a_held_writer(tmp_path, monkeypatch)
    with db:
        ticks = []
        asyncio.get_running_loop().call_later(0.05, lambda: ticks.append(released.is_set()))
        attach = asyncio.create_task(authority.attach(VIEWER, REF))
        await asyncio.sleep(0.3)
        assert ticks == [False], 'an unrelated 50 ms callback waited for the held writer'
        snapshot = await asyncio.wait_for(attach, 10)
        await asyncio.wait_for(drain, 10)
        # The snapshot still never shows the terminal row without its completion frame.
        complete = next(f['params']['seq'] for f in frames if f['params']['type'] == 'message.complete')
        if snapshot.handle.execution_state == 'idle':
            assert snapshot.last_sequence >= complete
        else:
            assert snapshot.last_sequence < complete


@pytest.mark.asyncio
async def test_pending_publication_during_a_contended_settlement_keeps_idle_after_completion(
        tmp_path, monkeypatch):
    db, authority, frames, released, drain = await _settling_behind_a_held_writer(tmp_path, monkeypatch)
    with db:
        ticks = []
        loop = asyncio.get_running_loop()
        # A loop-side publisher (submit/cancel/automation call this) racing the settlement.
        loop.call_later(0.02, authority._publish_pending, REF)
        loop.call_later(0.05, lambda: ticks.append(released.is_set()))
        await asyncio.sleep(0.3)
        assert ticks == [False], 'pending publication waited for the held writer'
        await asyncio.wait_for(drain, 10)
        kinds = _types(frames)
        start, complete = kinds.index(('message.start', None)), kinds.index(('message.complete', None))
        assert ('session.info', False) not in kinds[start:complete], kinds
        assert ('session.info', False) in kinds[complete:], kinds


@pytest.mark.asyncio
async def test_cancel_behind_a_held_writer_leaves_the_loop_free_and_still_settles_its_observer(
        tmp_path, monkeypatch):
    """``prompt.cancel`` / Delete on a queued card: its write waits on the writer, never on the loop."""
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        receipt = await authority.submit(ACTOR, Submission('queued', REF, {'text': 'queued'}, 'queue'))
        waiter = authority.waiters.setdefault(receipt.admission_id, asyncio.get_running_loop().create_future())
        released = threading.Event()
        writer = sqlite3.connect(str(tmp_path / 'state.db'), isolation_level=None, check_same_thread=False)
        writer.execute('BEGIN IMMEDIATE')

        def release():
            time.sleep(_HOLD_S)
            released.set()
            writer.rollback()
            writer.close()
        threading.Thread(target=release, daemon=True).start()
        ticks = []
        asyncio.get_running_loop().call_later(0.05, lambda: ticks.append(released.is_set()))
        cancel = asyncio.create_task(authority.cancel_queued(ACTOR, REF, receipt.admission_id))
        await asyncio.sleep(0.3)
        assert ticks == [False], 'an unrelated 50 ms callback waited for the held writer'
        cancelled = await asyncio.wait_for(cancel, 10)
        assert (cancelled.status, cancelled.outcome) == ('terminal', 'cancelled')
        assert waiter.done() and receipt.admission_id not in authority.cancel_obligations


def _hold_writer(tmp_path):
    """A second connection holds the state.db write lock for ``_HOLD_S`` (another process)."""
    released = threading.Event()
    writer = sqlite3.connect(str(tmp_path / 'state.db'), isolation_level=None, check_same_thread=False)
    writer.execute('BEGIN IMMEDIATE')

    def release():
        time.sleep(_HOLD_S)
        released.set()
        writer.rollback()
        writer.close()
    threading.Thread(target=release, daemon=True).start()
    return released


async def _loop_stays_free(tmp_path, operation):
    released = _hold_writer(tmp_path)
    ticks = []
    asyncio.get_running_loop().call_later(0.05, lambda: ticks.append(released.is_set()))
    task = asyncio.create_task(operation())
    await asyncio.sleep(0.3)
    assert ticks == [False], 'an unrelated 50 ms callback waited for the held writer'
    return await asyncio.wait_for(task, 10)


@pytest.mark.asyncio
async def test_submit_claim_and_discard_behind_a_held_writer_leave_the_loop_free(tmp_path, monkeypatch):
    """Admission, the FIFO claim and Discard wait on a held writer in a worker thread, tracked so
    retirement joins them; the claim's execution stamp still lands with its commit."""
    from gateway.session_runtime_workers import mutation_tasks
    from hermes_state_runtime import get_session_admission, recover_session_inputs, begin_runtime_epoch
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    live = authority.sessions['s']
    with db:
        receipt = await _loop_stays_free(
            tmp_path, lambda: authority.submit(ACTOR, Submission('work', REF, {'text': 'work'}, 'queue')))
        assert receipt.status == 'queued' and not mutation_tasks(authority)
        row, first = await _loop_stays_free(tmp_path, lambda: authority._claim_next(REF, live))
        assert row['admission_id'] == first['admission_id'] == receipt.admission_id
        assert live.event_stream.execution == {'authority_epoch': authority.epoch,
                                               'execution_generation': row['generation'],
                                               'admission_id': row['admission_id']}
        # The owner restarts mid-turn; the operator discards the now-unknown turn.
        authority.epoch = begin_runtime_epoch(db, instance_id='restart')
        recover_session_inputs(db, epoch=authority.epoch)
        live.event_stream.execution = {}
        resolved = await _loop_stays_free(
            tmp_path, lambda: authority.resolve_unknown(ACTOR, REF, receipt.admission_id, row['generation']))
        assert (resolved.status, resolved.outcome) == ('terminal', 'interrupted')
        assert get_session_admission(db, admission_id=receipt.admission_id)['status'] == 'terminal'
        assert not mutation_tasks(authority)


@pytest.mark.asyncio
async def test_input_admitted_while_an_idle_claim_is_in_flight_still_runs(tmp_path, monkeypatch):
    """The claim's transaction runs off-loop: an admission that commits after it read an empty
    FIFO must not be stranded by the drain exiting idle (its ``_schedule`` saw the drain alive)."""
    from gateway import session_authority as sa
    db, authority = _authority(tmp_path, monkeypatch)
    ran, read_empty, admitted = [], threading.Event(), threading.Event()
    claim = sa.claim_session_input

    def racing_claim(*args, **kwargs):
        row = claim(*args, **kwargs)
        if row is None and not read_empty.is_set():
            read_empty.set()
            admitted.wait(5)  # the submit commits and schedules while this claim reports idle
        return row
    monkeypatch.setattr(sa, 'claim_session_input', racing_claim)

    async def execute(owner, ref, row):
        ran.append(row['request_id'])
        owner.pending_results[row['admission_id']] = {'result': {'final_response': 'done'}, 'usage': {}}
        return 'done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    with db:
        authority._schedule(REF)
        assert await asyncio.to_thread(read_empty.wait, 5)
        # The API path's shape (admit_api_turn, then observe_api_turn's _schedule): it takes no
        # session mutation lock, so it can commit while the claim's transaction is in flight.
        from hermes_state_runtime import admit_session_input
        admit_session_input(db, epoch=authority.epoch, principal_id='api', session_id='s',
                            request_id='late', payload={'text': 'late'})
        authority._schedule(REF)
        admitted.set()
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert ran == ['late']


@pytest.mark.asyncio
async def test_admission_schedules_its_drain_in_the_admitting_callers_step(tmp_path, monkeypatch):
    """The ledger write runs off-loop, but the drain is scheduled only once the caller holds its
    receipt (the step in which messaging registers its delivery waiter), as it was when admission
    was synchronous: a drain started inside the write task could claim, run or pause the input
    before its waiter exists (a lost pause notice), and claim beside another drain's remote
    authorization await. A caller cancelled mid-commit still leaves its committed input a drain."""
    db, authority = _authority(tmp_path, monkeypatch)
    order = []

    async def drain():
        order.append('drain')

    def schedule(ref):
        order.append('schedule')
        asyncio.get_running_loop().create_task(drain())
    monkeypatch.setattr(authority, '_schedule', schedule)
    with db:
        receipt = await _submit(authority, 'first')
        order.append('caller')  # where messaging registers its delivery waiter
        await asyncio.sleep(0)
        assert order == ['schedule', 'caller', 'drain'] and receipt.status == 'queued'

        from gateway.session_runtime_workers import mutation_tasks
        order.clear()
        task = asyncio.ensure_future(_submit(authority, 'cancelled-mid-commit'))
        while not mutation_tasks(authority):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        while mutation_tasks(authority):
            await asyncio.sleep(0.01)
        # The commit's done-callback schedules the drain, and the drain task runs a step after
        # that: wait for the observable, not for one particular tick.
        async with asyncio.timeout(5):
            while order != ['schedule', 'drain']:
                await asyncio.sleep(0)
