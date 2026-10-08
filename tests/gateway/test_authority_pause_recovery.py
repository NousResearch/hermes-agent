"""Paused FIFO heads release observers and resume after their compute worker finishes."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_authority import LiveSession
from gateway.session_contract import Principal, SessionRef
from hermes_state_runtime import (RuntimeStoreError, admit_session_input, begin_runtime_epoch,
    get_session_admission, recover_session_inputs, register_worker_execution)


def session(owner):
    owner.db.create_session('s', source='telegram')
    owner.sessions['s'] = LiveSession(SessionSource(Platform.TELEGRAM, 'chat'), 'route')
    return SessionRef(owner.profile_id, 's')


@pytest.mark.asyncio
async def test_worker_block_releases_waiter_and_finish_wakes_queued_input(owner, monkeypatch):
    from gateway.session_worker import worker_request
    ref = session(owner)
    register_worker_execution(owner.db, epoch=owner.epoch, execution_id='worker', session_id='s',
        generation=0, kind='compute', adoption_secret='private')
    row, http_row = (admit_session_input(owner.db, epoch=owner.epoch, principal_id='human', session_id='s',
                                         request_id=name, payload={'text': name}) for name in ('queued', 'http'))
    waiter = owner.waiters[row['admission_id']] = asyncio.get_running_loop().create_future()
    owner.native_waiters.add(row['admission_id'])
    http_waiter = owner.waiters[http_row['admission_id']] = asyncio.get_running_loop().create_future()
    await owner._drain(ref)
    assert waiter.done(), 'a blocked FIFO left its messaging delivery waiter parked'
    with pytest.raises(RuntimeStoreError, match='session_busy'):
        await waiter
    # session_busy is transient: the HTTP caller keeps waiting for the real outcome, not a failure.
    assert not http_waiter.done()
    assert get_session_admission(owner.db, admission_id=row['admission_id'])['status'] == 'queued'
    seen = []
    async def execute(authority, ref, row):
        seen.append(row['request_id'])
        return 'done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    # Process identity is independent of this test's real receipt and scheduler path.
    monkeypatch.setattr('gateway.session_worker._claim', lambda *args: 'private')
    connection = SimpleNamespace(authority=owner, actor=Principal('human', owner.profile_id,
                                 frozenset({'worker:adopt'}), 'worker-transport'))
    await worker_request(connection, ref, dict(profile_id=owner.profile_id, session_id='s',
        execution_id='worker', generation=0, pid=1, birth=1, secret='private', epoch=owner.epoch,
        sequence=1, operation='execution.finish', payload={}), operation='persist')
    await owner.sessions['s'].task
    assert seen == ['queued', 'http']
    assert http_waiter.done() and http_waiter.exception() is None


@pytest.mark.asyncio
async def test_unknown_managed_execution_releases_all_observers_without_false_settlement(owner, monkeypatch):
    from gateway import session_managed_worker
    lost_execution = getattr(session_managed_worker, 'ManagedExecutionUnknown', asyncio.CancelledError)
    ref = session(owner)
    rows = [admit_session_input(owner.db, epoch=owner.epoch, principal_id='human', session_id='s',
        request_id=name, payload={'text': name}) for name in ('lost', 'follower')]
    waiters = [owner.waiters.setdefault(row['admission_id'], asyncio.get_running_loop().create_future()) for row in rows]
    async def execute(authority, ref, row):
        authority.epoch = begin_runtime_epoch(authority.db, instance_id='recovery')
        recover_session_inputs(authority.db, epoch=authority.epoch)
        raise lost_execution('worker lost')
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    await owner._drain(ref)
    for waiter in waiters:
        assert waiter.done()
        with pytest.raises(RuntimeStoreError, match='unknown_execution'):
            await waiter
    assert owner.sessions['s'].event_stream.execution == {}
    assert [get_session_admission(owner.db, admission_id=row['admission_id'])['status'] for row in rows] == ['unknown', 'queued']
