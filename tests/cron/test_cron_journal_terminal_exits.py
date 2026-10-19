"""Bug class: a canonical cron firer books its own terminal ledger row on a non-completion exit
(ownership lost, shutdown interrupt, crashed tail) but leaves the admission journal behind;
receipt recovery then replays the owner's result: a second output save, delivery and run mark.
"""
from contextlib import asynccontextmanager
import json

import pytest

from cron import executions, jobs, scheduler, scheduler_authority
from cron.scheduler_bookkeeping import finish_interrupted_run
from hermes_cli import gateway_client


def _ownership_lost(job, fire, journal):
    scheduler._record_fire_ownership_lost(job, None, fire)


def _shutdown_interrupted(job, fire, journal):
    finish_interrupted_run(job, fire, None)


def _crashed_tail(job, fire, journal):
    # run_one_job's generic exception tail: it books the row failed itself.
    assert scheduler._run_one_job_body(dict(job, execution_id=fire)) is False


@pytest.mark.parametrize('firer_exit', [_ownership_lost, _shutdown_interrupted, _crashed_tail],
                         ids=['ownership-lost', 'shutdown-interrupted', 'crashed-tail'])
def test_firer_booked_exit_is_not_replayed_by_receipt_recovery(tmp_path, monkeypatch, firer_exit):
    calls = []

    class Peer:
        async def rpc(self, method, **params):
            calls.append(method)
            return {'status': 'terminal', 'job': job, 'result': [True, 'document', 'answer', None]}

    @asynccontextmanager
    async def connected():
        yield Peer()
    monkeypatch.setattr(gateway_client, 'connect_gateway', connected)

    def run_job(job, **kwargs):
        raise RuntimeError('the firing tail crashed after the owner finished')
    monkeypatch.setattr(scheduler, 'run_job', run_job)
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')
        fire = executions.create_execution(job['id'], source='test')['id']
        if firer_exit is not _crashed_tail:
            executions.mark_execution_running(fire)
        journal = scheduler_authority.journal_path(job['id'], fire)
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(json.dumps({'params': {'job_id': job['id'], 'request_id': fire, 'extra_prompt': None},
                                       'receipt': {'session_id': 's', 'admission_id': 'a'}}))
        firer_exit(dict(jobs.get_job(job['id'])), fire, journal)
        assert executions.get_execution(fire)['status'] == 'failed'
        booked = jobs.get_job(job['id'])
        scheduler_authority.reconcile_pending()
        after = jobs.get_job(job['id'])
        assert calls == []
        assert not list(jobs._job_output_dir(job['id']).glob('*.md'))
        assert (after['last_run_at'], after['last_status'], after['repeat']) == (
            booked['last_run_at'], booked['last_status'], booked['repeat'])
        assert executions.get_execution(fire)['status'] == 'failed'
        assert not journal.exists()
