"""A busy cron ledger on the firer's informational progress stamp must not cost a live run its
supervision: the run completes, the job is not paused and the owner gets no failure notice."""
import asyncio
import functools
import sqlite3
from contextlib import asynccontextmanager

from cron import executions, jobs, scheduler
from cron.scheduler_worker_failure import record_unknown_worker_outcome
from hermes_cli import gateway_client, sqlite_util


def _fire(tmp_path, monkeypatch, status):
    """Fire one job in-process through the canonical observer against an owner whose
    ``cron.status`` answers are scripted by ``status``; return the body's verdict, the job, its
    execution row, the status poll count and the owner failure notices the dispatcher sent."""
    db = tmp_path / 'executions.db'
    monkeypatch.setattr(executions, 'EXECUTIONS_FILE', db)
    # The ledger's 5 s busy timeout, shortened so a held lock fails fast.
    monkeypatch.setattr(sqlite_util, 'open_db',
                        functools.partial(sqlite_util.open_db, busy_timeout_ms=50))
    polls = []

    class Peer:
        async def rpc(self, method, **params):
            if method == 'cron.submit':
                return {'session_id': 'cron-fire', 'admission_id': 'admit'}
            polls.append(method)
            return status(len(polls), db)

    @asynccontextmanager
    async def connected():
        yield Peer()

    real_sleep = asyncio.sleep

    async def no_wait(delay):
        await real_sleep(0)

    notices = []
    monkeypatch.setattr(gateway_client, 'connect_gateway', connected)
    monkeypatch.setattr(asyncio, 'sleep', no_wait)
    monkeypatch.setattr(scheduler, '_deliver_crash_failure',
                        lambda job, error, **kw: (notices.append(error), (None, 'delivered'))[1])
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')
        claimed = jobs.claim_job_for_fire(job['id'], manual=True, return_job=True)
        assert isinstance(claimed, dict)
        claimed['execution_id'] = executions.create_execution(job['id'], source='test')['id']
        ok = scheduler._run_one_job_body(claimed)
        # What the dispatcher does once the worker exits: notify only for an unknown outcome.
        record_unknown_worker_outcome(claimed)
        return (ok, jobs.get_job(job['id']), executions.get_execution(claimed['execution_id']),
                len(polls), notices)


def test_locked_ledger_on_progress_stamp_keeps_supervising_the_run(tmp_path, monkeypatch, caplog):
    holder = []
    stamp = executions.touch_execution_progress

    def touch(execution_id):
        # Another writer (a heavy test run on the same host) holds the ledger's write lock across
        # the firer's progress stamp, then lets go.
        try:
            return stamp(execution_id)
        finally:
            holder.pop().rollback()

    def status(n, db):
        if n == 1:
            conn = sqlite3.connect(db, timeout=0, isolation_level=None, check_same_thread=False)
            conn.execute('BEGIN IMMEDIATE')
            holder.append(conn)
            return {'status': 'started', 'progress_at': 'p1'}
        return {'status': 'terminal', 'result': [True, 'document', 'answer', None]}

    monkeypatch.setattr(executions, 'touch_execution_progress', touch)
    ok, job, row, polls, notices = _fire(tmp_path, monkeypatch, status)
    assert holder == []
    assert ok is True
    assert polls == 2
    assert job['state'] != 'paused'
    assert row['status'] == 'completed'
    assert notices == []
    assert 'progress stamp skipped: database is locked' in caplog.text


def test_owner_lost_mid_run_still_pauses_and_notifies(tmp_path, monkeypatch):
    def status(n, db):
        raise gateway_client.GatewayClientError('Gateway disconnected')

    ok, job, row, _, notices = _fire(tmp_path, monkeypatch, status)
    assert ok is False
    assert job['state'] == 'paused'
    assert row['status'] == 'unknown'
    assert len(notices) == 1 and 'Gateway disconnected' in notices[0]
