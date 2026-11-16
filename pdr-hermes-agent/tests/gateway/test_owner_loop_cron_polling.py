"""A running cron job's owner-side polling never blocks the gateway event loop on storage.

While an agent cron job runs, its in-process firer polls the owner (``cron/scheduler_authority.py``
``observe`` -> ``gateway/session_cron.py`` ``operation('status')``) every <= 2 s, and journals its
receipt with an fsynced write (``save``). Both used to run on the owner loop: on slow storage
(a network filesystem holding SQLite's WAL ``-shm`` lock for seconds) every poll stopped every
socket, timer, Stop and liveness probe the gateway serves for as long as the read took.

Real GatewayRunner + SessionAuthority + SessionDB; storage is slowed by wrapping the store's own
choke points (and the journal write), so the stall is measured as loop-tick lag, not inferred.
"""
import asyncio
import threading
import time

import pytest

SLOW_S = 2.0  # one slow storage call; the report measured 5-58 s stalls on NFS
MAX_LAG_S = 0.2


def _ticker(gaps):
    async def tick():
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.02)  # health: allow HX013 -- the ticker's period is the owner-loop stall measurement
            now = time.monotonic()
            gaps.append(now - last)
            last = now
    return asyncio.create_task(tick())


async def _owner(tmp_path, monkeypatch, run_job):
    from cron import jobs, scheduler
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr('hermes_state.DEFAULT_DB_PATH', tmp_path / 'state.db')
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {'model': {}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(jobs, 'get_job', lambda jid: {'id': jid, 'prompt': 'report', 'model': 'm', 'name': 'job'})
    monkeypatch.setattr(scheduler, 'run_job', run_job)
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=False, sessions_dir=tmp_path / 'sessions'))
    # The owner binds itself as this home's in-process cron owner (session_cron.bind_owner).
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    return runner, authority


def _slow_store(monkeypatch, slow):
    """Every SessionDB read and write waits ``SLOW_S`` while ``slow`` is set, on whatever thread."""
    from hermes_state import SessionDB
    read, write = SessionDB._read_ctx, SessionDB._execute_write

    def _read_ctx(self, *args, **kwargs):
        if slow.is_set():
            time.sleep(SLOW_S)  # health: allow HX013 -- the slow storage read IS the stall under test
        return read(self, *args, **kwargs)

    def _execute_write(self, *args, **kwargs):
        if slow.is_set():
            time.sleep(SLOW_S)  # health: allow HX013 -- the slow storage write IS the stall under test
        return write(self, *args, **kwargs)
    monkeypatch.setattr(SessionDB, '_read_ctx', _read_ctx)
    monkeypatch.setattr(SessionDB, '_execute_write', _execute_write)


@pytest.mark.asyncio
@pytest.mark.parametrize('cancelled', [False, True], ids=['status-poll', 'cancel-poll'])
async def test_cron_owner_polls_keep_the_loop_ticking_on_slow_storage(tmp_path, monkeypatch, cancelled):
    from cron.scheduler_authority import run_canonical_job
    running, release = threading.Event(), threading.Event()

    def run_job(job, *, extra_prompt, execution_id, cancel_event):
        running.set()
        release.wait(30)
        return (True, 'doc', 'answer', None)
    runner, _authority = await _owner(tmp_path, monkeypatch, run_job)
    slow = threading.Event()
    _slow_store(monkeypatch, slow)
    cancel = threading.Event()
    firer = asyncio.create_task(asyncio.to_thread(
        run_canonical_job, {'id': 'job'}, execution_id='fire', cancel_event=cancel))
    gaps = []
    try:
        assert await asyncio.to_thread(running.wait, 10), 'the cron job never started on the owner'
        await asyncio.sleep(2.5)  # health: allow HX013 -- the firer's status backoff reaches its 2 s cap
        if cancelled:
            cancel.set()  # the firer now sends cancel + status on every poll
        slow.set()
        tick = _ticker(gaps)
        await asyncio.sleep(SLOW_S * 2.5)  # health: allow HX013 -- the slow-storage window the ticker measures
        tick.cancel()
    finally:
        slow.clear()
        release.set()
        result = await asyncio.wait_for(firer, 30)
        await asyncio.to_thread(runner.session_store.close_all_db_handles)
    assert result[0] is True or cancelled, result
    assert gaps and max(gaps) < MAX_LAG_S, f'owner loop stalled {max(gaps):.2f}s behind a cron poll'


@pytest.mark.asyncio
async def test_cron_receipt_journal_fsync_keeps_the_loop_ticking(tmp_path, monkeypatch):
    """``observe``'s journal ``save()`` (atomic write + file and directory fsync) runs off the loop."""
    import utils
    from cron.scheduler_authority import run_canonical_job

    def run_job(job, *, extra_prompt, execution_id, cancel_event):
        return (True, 'doc', 'answer', None)
    runner, _authority = await _owner(tmp_path, monkeypatch, run_job)
    # Warm the fire path (cold imports of the submit/execute modules are not the write under test).
    await asyncio.wait_for(asyncio.to_thread(run_canonical_job, {'id': 'job'}, execution_id='warm'), 30)
    real_write, slow_writes = utils.atomic_json_write, []

    def slow_journal_write(path, *args, **kwargs):
        if '/cron/admissions/' in str(path):
            slow_writes.append(threading.get_ident())
            time.sleep(SLOW_S / 2)  # health: allow HX013 -- a slow (NFS) fsync IS the stall under test
        return real_write(path, *args, **kwargs)
    monkeypatch.setattr(utils, 'atomic_json_write', slow_journal_write)
    gaps = []
    tick = _ticker(gaps)
    try:
        result = await asyncio.wait_for(asyncio.to_thread(run_canonical_job, {'id': 'job'}, execution_id='fire'), 30)
    finally:
        tick.cancel()
        await asyncio.to_thread(runner.session_store.close_all_db_handles)
    assert result == (True, 'doc', 'answer', None)
    assert len(slow_writes) >= 2, 'the journal was not written before submit and after the receipt'
    assert max(gaps) < MAX_LAG_S, f'owner loop stalled {max(gaps):.2f}s behind the cron journal write'
