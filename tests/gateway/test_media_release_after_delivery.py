"""Terminal media collection cannot delay replies, including recovered settlements."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import session_ingress, session_results, session_settlement_recovery as recovery
from gateway.session_runtime_workers import mutation_tasks
from hermes_state_runtime import get_session_admission
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['normal', 'republish', 'resolved', 'send_failure', 'waiter', 'publish_failure'])
async def test_delivery_precedes_slow_media_collection(tmp_path, monkeypatch, path):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    started, allow_cleanup = threading.Event(), threading.Event()
    order, releases = [], []
    release = recovery._release_media

    def slow_release(owner, row):
        releases.append(row['admission_id'])
        order.append('cleanup')
        started.set()
        assert allow_cleanup.wait(10), 'test did not release the cleanup worker'
        release(owner, row)

    monkeypatch.setattr(recovery, '_release_media', slow_release)

    async def execute(owner, ref, row):
        owner.pending_results[row['admission_id']] = {'result': {'final_response': 'reply'}, 'usage': {}}
        event = SimpleNamespace(message_id=row['admission_id'])
        owner.pending_deliveries[row['admission_id']] = (object(), event, 'route', 'reply', None)
        return 'reply'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)

    async def send(adapter, event, key, response):
        assert get_session_admission(db, admission_id=event.message_id)['status'] == 'terminal'
        order.append('send')
        if path == 'send_failure':
            raise OSError('transport unavailable')

    monkeypatch.setattr(session_ingress, 'deliver_response', send)
    finish = session_results.finish_result
    if path in {'republish', 'resolved'}:
        monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', ())

        def fail(*args, **kwargs):
            if path == 'republish':
                finish(*args, **kwargs)
            raise OSError('settlement response unavailable')

        monkeypatch.setattr(session_results, 'finish_result', fail)

    if path == 'publish_failure':
        def fail_publish(*args):
            raise OSError('viewer disconnected')
        monkeypatch.setattr(recovery, '_publish_completion', fail_publish)

    with db:
        receipt = await _submit(authority, 'photo')
        if path == 'waiter':
            authority.native_waiters.add(receipt.admission_id)
            waiter = authority.waiters[receipt.admission_id] = asyncio.get_running_loop().create_future()

            async def send_waiter():
                assert await waiter == 'reply'
                order.append('send')

            observer = asyncio.create_task(send_waiter())
        drain = asyncio.create_task(authority._drain(REF))
        operation = drain
        try:
            if path == 'resolved':
                await asyncio.wait_for(drain, 5)
                row = get_session_admission(db, admission_id=receipt.admission_id)
                assert row['status'] == 'unknown' and not releases
                operation = asyncio.create_task(authority.resolve_unknown_as_owner(
                    REF, receipt.admission_id, row['generation']))
            await async_wait_until(lambda: bool(order) or operation.done(), timeout=5, what='delivery or cleanup')
            if not order:
                await operation
            assert order
            assert order[0] == 'send', 'media cleanup ran before the reply was delivered'
            await async_wait_until(started.is_set, timeout=5, what='background media cleanup')
            await asyncio.wait_for(operation, 5)
            assert mutation_tasks(authority), 'cleanup must keep ownership until its worker exits'
            from gateway.run_runtime import _retire_profile_authority
            assert not await _retire_profile_authority(authority, timeout=0)
        finally:
            allow_cleanup.set()
            await asyncio.gather(operation, drain, return_exceptions=True)
            await async_wait_until(lambda: not mutation_tasks(authority), timeout=5, what='cleanup retirement')
            if path == 'waiter':
                await observer
        assert releases == [receipt.admission_id]
        assert await _retire_profile_authority(authority, timeout=5)

@pytest.mark.asyncio
@pytest.mark.parametrize('captured', [True, False])
async def test_resolution_racing_recovery_releases_once(tmp_path, monkeypatch, captured):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    entered, resume = threading.Event(), threading.Event()
    released = []
    mark = recovery._mark_unknown

    def held_mark(*args):
        result = mark(*args)
        entered.set()
        assert resume.wait(10)
        return result

    def fail(*args, **kwargs):
        raise OSError('storage unavailable')

    async def execute(owner, ref, row):
        owner.pending_results[row['admission_id']] = {'result': {'final_response': 'reply'}, 'usage': {}}
        return 'reply'

    monkeypatch.setattr(recovery, '_mark_unknown', held_mark)
    monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', ())
    monkeypatch.setattr(session_results, 'finish_result', fail)
    monkeypatch.setattr(recovery, '_release_media', lambda owner, row: released.append(row['admission_id']))
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    with db:
        receipt = await _submit(authority, 'photo')
        drain = asyncio.create_task(authority._drain(REF))
        try:
            await async_wait_until(entered.is_set, timeout=5, what='recovery stamp')
            row = get_session_admission(db, admission_id=receipt.admission_id)
            if not captured:
                authority.pending_results.pop(receipt.admission_id)
            await authority.resolve_unknown_as_owner(REF, receipt.admission_id, row['generation'])
            assert released == [], 'resolution must leave cleanup to the live recovery owner'
        finally:
            resume.set()
            await asyncio.wait_for(drain, 5)
            await async_wait_until(lambda: not mutation_tasks(authority), timeout=5, what='recovery cleanup')
        assert released == [receipt.admission_id]


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['commit', 'delivery'])
async def test_cancelled_drain_keeps_delivery_and_collection_owned(tmp_path, monkeypatch, phase):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    blocked, resume = threading.Event(), threading.Event()
    collecting, finish_cleanup = threading.Event(), threading.Event()
    releases, sent = [], []
    original = recovery.commit_and_publish

    def commit(*args):
        if phase == 'commit':
            blocked.set()
            assert resume.wait(10)
        return original(*args)

    def release(owner, row):
        releases.append(row['admission_id'])
        collecting.set()
        assert finish_cleanup.wait(10)

    async def execute(owner, ref, row):
        owner.pending_deliveries[row['admission_id']] = (object(), object(), 'route', 'reply', None)
        return 'reply'

    async def send(*args):
        if phase == 'delivery':
            blocked.set()
            await async_wait_until(resume.is_set, timeout=5, what='delivery resume')
        sent.append('reply')

    monkeypatch.setattr(recovery, 'commit_and_publish', commit)
    monkeypatch.setattr(recovery, '_release_media', release)
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    monkeypatch.setattr(session_ingress, 'deliver_response', send)
    with db:
        receipt = await _submit(authority, 'photo')
        drain = asyncio.create_task(authority._drain(REF))
        try:
            await async_wait_until(blocked.is_set, timeout=5, what='commit or delivery blocked')
            drain.cancel()
            await asyncio.sleep(0)
            assert mutation_tasks(authority) and not releases
            resume.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(drain, 5)
            await async_wait_until(collecting.is_set, timeout=5, what='owned collection')
            assert sent == ['reply'] and releases == [receipt.admission_id]
            assert mutation_tasks(authority)
        finally:
            resume.set()
            finish_cleanup.set()
            await asyncio.gather(drain, return_exceptions=True)
            await async_wait_until(lambda: not mutation_tasks(authority), timeout=5, what='collection complete')
        assert releases == [receipt.admission_id]


@pytest.mark.asyncio
async def test_shutdown_joins_cleanup_created_after_its_initial_snapshot(monkeypatch):
    from gateway import run_runtime
    from gateway import session_runtime_workers
    from gateway.session_runtime_workers import track_mutation

    joining = asyncio.Event()
    original_join = session_runtime_workers.join_authority_work

    async def join(*args):
        joining.set()
        await original_join(*args)

    monkeypatch.setattr(session_runtime_workers, 'join_authority_work', join)
    entered, resume_delivery, collecting, finish_cleanup = (asyncio.Event() for _ in range(4))
    authority = SimpleNamespace(sessions={}, pending_stops={}, adopted={})
    runner = SimpleNamespace(session_authority=authority)
    authority.runner = runner

    async def collect():
        collecting.set()
        await finish_cleanup.wait()

    async def deliver():
        entered.set()
        await resume_delivery.wait()
        track_mutation(authority, collect())

    track_mutation(authority, deliver())
    await entered.wait()
    shutdown = asyncio.create_task(run_runtime.settle_gateway_runtime(runner, budget=5 + run_runtime.STOP_TAIL_RESERVE_SECONDS))
    try:
        await asyncio.sleep(0)  # shutdown snapshots the delivery task, not the collector
        resume_delivery.set()
        await collecting.wait()
        await joining.wait()
        await asyncio.sleep(0)
        assert not shutdown.done(), 'shutdown skipped cleanup spawned by a completing delivery'
        finish_cleanup.set()
        await asyncio.wait_for(shutdown, 5)
        assert not mutation_tasks(authority)
    finally:
        resume_delivery.set()
        finish_cleanup.set()
        await asyncio.gather(shutdown, return_exceptions=True)

