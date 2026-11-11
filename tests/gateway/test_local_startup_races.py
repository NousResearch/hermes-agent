"""Re-entry, cancellation and viewer races across local startup preparation."""
import asyncio
import threading

import pytest

from tests.gateway.test_local_session_orphan_end import _attach, _create, _owner
from tests.waiting import async_wait_until


def _pause_prepared(monkeypatch):
    from gateway import session_local_startup as startup
    entered, release = threading.Event(), threading.Event()
    original = startup._prepare

    def prepare(*args):
        result = original(*args)
        entered.set()
        assert release.wait(10), 'test did not release preparation'
        return result

    monkeypatch.setattr(startup, '_prepare', prepare)
    return entered, release


@pytest.mark.asyncio
@pytest.mark.parametrize('race', ['attach', 'close', 'retire', 'adapter', 'route', 'activity'])
async def test_suspended_prepare_rechecks_live_state(tmp_path, monkeypatch, race):
    from gateway.config import Platform
    from gateway.session_local_startup import recover_local_sessions_async
    from gateway.session_viewer_close import _PendingClose, _persist_close
    from gateway.session_runtime_workers import mutation_tasks

    from datetime import timedelta
    authority, _ = await _owner(tmp_path, monkeypatch)
    sid = _create(authority).session_id
    other = _create(authority, 'unrelated orphan').session_id
    route = authority.sessions[sid].route
    original_live = authority.sessions[sid]
    close = _PendingClose('cli_close')
    _persist_close(authority, sid, close.record)
    # Recover an unrestored receipt; an attaching client is allowed to create the adapter.
    authority.sessions.clear()
    authority.runner.adapters.pop(Platform.LOCAL)
    entered, release = _pause_prepared(monkeypatch)
    scheduled = []
    monkeypatch.setattr(authority, '_schedule', lambda ref: scheduled.append(ref.session_id))
    task = asyncio.create_task(recover_local_sessions_async(authority, schedule=True))
    viewer = None
    try:
        await async_wait_until(entered.is_set)
        newer = _PendingClose('new_session')

        async def attach():
            nonlocal viewer
            viewer, _ = await _attach(authority, sid)

        def close_new():
            authority.pending_closes[sid] = newer
            _persist_close(authority, sid, newer.record)

        def replace_route():
            authority.db.create_session('foreign', source='cli')
            authority.db.append_message('foreign', 'user', 'not a stray')
            authority.runner.session_store._entries[route].session_id = 'foreign'

        def activity():
            authority.runner.session_store._entries[route].yolo = True
            authority.runner.session_store._entries[route].updated_at += timedelta(seconds=1)

        action = {
            'attach': attach, 'close': close_new, 'route': replace_route, 'activity': activity,
            'retire': lambda: setattr(authority, 'retiring', True),
            'adapter': lambda: authority.runner.adapters.__setitem__(Platform.LOCAL, object()),
        }[race]()
        if asyncio.iscoroutine(action):
            await action
        release.set()
        await asyncio.wait_for(task, 10)
        await asyncio.gather(*mutation_tasks(authority))
        if race == 'attach':
            assert authority.sessions[sid].subscribers
            assert other in authority.sessions and other in authority.orphan_closes
            from gateway.session_viewer_close import CLOSE_PREFIX
            assert authority.db.get_meta(CLOSE_PREFIX + sid) is None
            assert sid not in authority.pending_closes and sid not in authority.orphan_closes
        elif race == 'close':
            assert authority.pending_closes[sid] is newer
        elif race in ('retire', 'adapter', 'route'):
            assert sid not in authority.sessions and scheduled == []
        else:
            assert sid in authority.sessions
            assert authority.runner.session_store._entries[route].yolo is True
            assert authority.sessions[sid] is not original_live
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        for handle in authority.orphan_closes.values():
            handle.cancel()
        if viewer:
            await viewer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('initial', [False, True])
async def test_cancelled_compression_preparation_retains_ownership_until_worker_finishes(tmp_path, monkeypatch, initial):
    from gateway.session_local_startup import recover_local_sessions_async
    from gateway.session_runtime_workers import mutation_tasks, join_authority_work
    from hermes_state_local import local_receipt
    from tests.gateway.test_local_receipt_untracked_compression import _compact_like_0_22_2
    import hermes_state_local_lineage as lineage

    authority, _ = await _owner(tmp_path, monkeypatch)
    sid = _create(authority).session_id
    _compact_like_0_22_2(authority.db, sid, 'child')
    authority.sessions.clear()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = lineage.adopt_untracked_compression

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(lineage, 'adopt_untracked_compression', held)
    task = asyncio.create_task(recover_local_sessions_async(authority, initial=initial))
    try:
        await async_wait_until(entered.is_set)
        task.cancel()
        await asyncio.sleep(0)
        if initial:
            assert not task.done()
            task.cancel()  # repeated cancellation must not free the unpublished owner's DB
            for _ in range(4):
                await asyncio.sleep(0)
            assert not task.done()
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert mutation_tasks(authority)
            with pytest.raises(TimeoutError, match='ownership retained'):
                await join_authority_work(authority, timeout=0)
        release.set()
        await async_wait_until(finished.is_set)
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*mutation_tasks(authority))
        assert task.cancelled()
        assert local_receipt(authority.db, sid)['entry']['session_id'] == 'child'
        assert sid not in authority.sessions
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*mutation_tasks(authority))


@pytest.mark.asyncio
async def test_compression_committed_between_adoption_scan_and_prepare_is_caught_up(tmp_path, monkeypatch):
    from gateway import session_local_startup as startup
    from hermes_state_local import local_receipt
    from tests.gateway.test_local_receipt_untracked_compression import _compact_like_0_22_2

    authority, _ = await _owner(tmp_path, monkeypatch)
    sid = _create(authority).session_id
    original, calls = startup._prepare, []

    def intervening(*args):
        if not calls:
            _compact_like_0_22_2(authority.db, sid, 'late-child')
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(startup, '_prepare', intervening)
    await startup.recover_local_sessions_async(authority)
    assert sid in authority.sessions
    assert local_receipt(authority.db, sid)['entry']['session_id'] == 'late-child'


@pytest.mark.asyncio
async def test_reattach_then_live_handoff_does_not_resurrect_prepared_close(tmp_path, monkeypatch):
    from gateway.session_local_startup import recover_local_sessions_async
    from gateway.session_viewer_close import CLOSE_PREFIX, _PendingClose, _persist_close
    authority, _ = await _owner(tmp_path, monkeypatch)
    sid = _create(authority).session_id
    _persist_close(authority, sid, _PendingClose('cli_close').record)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    entered, release = _pause_prepared(monkeypatch)
    task = asyncio.create_task(recover_local_sessions_async(authority, schedule=True))
    try:
        await async_wait_until(entered.is_set)
        viewer, resumed = await _attach(authority, sid)
        reply = await viewer.dispatch({'id': 2, 'method': 'session.detach', 'params': {
            'session_id': sid, 'subscription_id': resumed['subscription_id']}})
        assert 'result' in reply
        assert not authority.sessions[sid].subscribers
        assert authority.db.get_meta(CLOSE_PREFIX + sid) is None
        release.set()
        await task
        assert sid not in authority.pending_closes
        assert sid not in authority.orphan_closes
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        for handle in authority.orphan_closes.values():
            handle.cancel()


@pytest.mark.asyncio
async def test_async_startup_rearms_persisted_close_on_a_legacy_adopted_session(tmp_path, monkeypatch):
    from gateway.session_local_startup import recover_local_sessions_async
    from hermes_state_runtime import admit_session_input, cancel_session_input
    from tests.gateway.test_viewer_close_lifecycle import _call, _open, _owner as native_owner, _restarted_owner

    db, authority, _ = await native_owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    row = admit_session_input(db, epoch=authority.epoch, principal_id='native', session_id='chat',
                              request_id='r1', payload={'text': 'mid-turn'})
    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='cli_close')
    cancel_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'])
    db2, restarted, calls = await _restarted_owner(tmp_path, monkeypatch, db)
    await recover_local_sessions_async(restarted, schedule=True)
    await async_wait_until(lambda: len(calls) == 2, timeout=3)
    assert db2.get_session('chat')['end_reason'] == 'cli_close'
