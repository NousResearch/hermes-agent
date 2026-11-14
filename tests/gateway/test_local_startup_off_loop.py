"""Local receipt recovery must not stall the owner loop, including before publication."""
import asyncio
import threading

import pytest

from tests.gateway.test_local_session_orphan_end import _owner


def _empty_cron_receipts(authority, monkeypatch, count=12):
    import time
    from gateway.session_cron import _actor, _create
    from hermes_state_local import commit_local_session
    monkeypatch.setattr('cron.jobs.get_job', lambda jid: {'id': jid, 'name': jid, 'prompt': 'synthetic', 'model': 'fixture-model'})
    ids = []
    for i in range(count):
        receipt, _ = _create(authority, _actor(authority, None), {
            'job_id': 'synthetic', 'request_id': str(i), 'extra_prompt': None})
        commit_local_session(authority.db, epoch=authority.epoch, receipt=receipt)
        ids.append(receipt['session_id'])
    authority.db._execute_write(lambda conn: conn.executemany(
        'UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?',
        [(time.time() - 8 * 3600, time.time() - 8 * 3600, sid) for sid in ids]))
    return ids


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['cold', 'late', 'late-idle'])
async def test_local_backlog_read_leaves_owner_loop_responsive(tmp_path, monkeypatch, phase):
    from gateway import session_local_recovery as recovery
    from gateway.run_runtime import recover_gateway_native_sessions

    first, _ = await _owner(tmp_path, monkeypatch)
    ids = _empty_cron_receipts(first, monkeypatch)
    assert all(first.db.get_session(sid)['message_count'] == 0 for sid in ids)
    owner_thread = threading.get_ident()
    entered, release, control = threading.Event(), threading.Event(), threading.Event()
    observed = []
    from gateway import session_viewer_close as closing
    target, name = (closing, '_idle_s') if phase == 'late-idle' else (recovery, 'local_receipt_ids')
    original = getattr(target, name)

    def stalled(authority, *args):
        observed.append(threading.get_ident())
        if not entered.is_set():
            entered.set()
            release.wait(3)
        return original(authority, *args)

    async def control_task():
        await asyncio.to_thread(entered.wait, 5)
        control.set()
        release.set()

    async def nothing(*args, **kwargs):
        return {}

    monkeypatch.setattr(target, name, stalled)
    monkeypatch.setattr('gateway.session_hosted_service.ensure_hosted_service', nothing)
    monkeypatch.setattr('gateway.platforms.webhook_ingress.recover_webhook_finalizations', nothing)
    monkeypatch.setattr('gateway.run_runtime.recover_profile_native_sessions', nothing)
    heartbeat = asyncio.create_task(control_task())
    try:
        if phase == 'cold':
            restored, _ = await _owner(tmp_path, monkeypatch)
            assert len(restored.sessions) == 12
        else:
            await recover_gateway_native_sessions(first.runner)
        assert control.is_set(), 'local receipt scan blocked the owner control task'
        assert observed and all(t != owner_thread for t in observed)
    finally:
        release.set()
        await heartbeat
        for handle in first.orphan_closes.values():
            handle.cancel()


@pytest.mark.asyncio
async def test_exhausted_reader_fallback_keeps_idle_reads_nonnested(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from gateway.session_local_startup import recover_local_sessions_async
    authority, _ = await _owner(tmp_path, monkeypatch)
    ids = _empty_cron_receipts(authority, monkeypatch, count=3)
    db, reads, local = authority.db, [], threading.local()
    monkeypatch.setattr(db, '_checkout_read_conn', lambda: None)
    original = db._read_ctx

    @contextmanager
    def guarded(**kwargs):
        assert not getattr(local, 'reading', False), 'nested fallback read would deadlock the writer lock'
        local.reading = True
        reads.append(threading.get_ident())
        try:
            with original(**kwargs) as conn:
                yield conn
        finally:
            local.reading = False

    monkeypatch.setattr(db, '_read_ctx', guarded)
    try:
        await asyncio.wait_for(recover_local_sessions_async(authority, schedule=True), 10)
        assert set(authority.orphan_closes) == set(ids)
        assert reads and threading.get_ident() not in reads
    finally:
        for handle in authority.orphan_closes.values():
            handle.cancel()
