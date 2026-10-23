"""Receipt recovery: a ``missing`` admission is a definite non-admission (booked failed, never a
pause), and a verified terminal receipt lifts the pause its own ``CronExecutionUnknown`` caused."""
from contextlib import asynccontextmanager
import json

from cron import executions, jobs, scheduler, scheduler_authority
from hermes_cli import gateway_client


def _peer(monkeypatch, answer):
    class Peer:
        async def rpc(self, method, **params):
            return answer(method)

    @asynccontextmanager
    async def connected():
        yield Peer()
    monkeypatch.setattr(gateway_client, 'connect_gateway', connected)


def _foreign_journal(job, fire):
    with executions._transaction() as conn:
        conn.execute('UPDATE executions SET process_id=?,pid=? WHERE id=?', ('old-owner', 999999, fire))
    journal = scheduler_authority.journal_path(job['id'], fire)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({'params': {'job_id': job['id'], 'request_id': fire, 'extra_prompt': None},
                                   'receipt': None}))
    return journal


def test_missing_admission_is_booked_failed_not_paused(tmp_path, monkeypatch):
    """The firer died after writing its journal but before the owner admitted the fire."""
    _peer(monkeypatch, lambda method: {'status': 'missing', 'result': None})
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')
        fire = executions.create_execution(job['id'], source='test')['id']
        executions.mark_execution_running(fire)
        journal = _foreign_journal(job, fire)
        scheduler_authority.reconcile_pending()
        saved = jobs.get_job(job['id'])
        assert saved['state'] != 'paused' and saved['enabled'] is True
        assert saved['last_status'] == 'error'
        assert executions.get_execution(fire)['status'] == 'failed'
        assert not journal.exists()


def _unknown_fire(monkeypatch, job, fire):
    """The firer journals the fire, then loses its observer: CronExecutionUnknown pauses the job."""
    journal = scheduler_authority.journal_path(job['id'], fire)

    def run_job(job, **kwargs):
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(json.dumps({'params': {'job_id': job['id'], 'request_id': fire,
                                                  'extra_prompt': None}, 'receipt': None}))
        raise scheduler_authority.CronExecutionUnknown('Cron execution unverified: observer dropped')
    monkeypatch.setattr(scheduler, 'run_job', run_job)
    assert scheduler._run_one_job_body(dict(jobs.get_job(job['id']), execution_id=fire)) is False
    assert jobs.get_job(job['id'])['state'] == 'paused'
    return journal


def test_terminal_receipt_lifts_the_unknown_pause_it_caused(tmp_path, monkeypatch):
    """Once the owner's receipt proves the run terminal, the recurring schedule resumes by itself;
    an operator's own later pause is never lifted by a receipt."""
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')
        fire = executions.create_execution(job['id'], source='test')['id']
        journal = _unknown_fire(monkeypatch, job, fire)
        _peer(monkeypatch, lambda method: {'status': 'terminal', 'job': job,
                                           'result': [True, 'document', 'answer', None]})
        scheduler_authority.reconcile_pending()
        saved = jobs.get_job(job['id'])
        assert (saved['state'], saved['enabled'], saved['paused_reason']) == ('scheduled', True, None)
        assert executions.get_execution(fire)['status'] == 'completed'
        assert not journal.exists()

        second = executions.create_execution(job['id'], source='test')['id']
        _unknown_fire(monkeypatch, job, second)
        jobs.resume_job(job['id'])
        jobs.pause_job(job['id'], reason='operator hold')
        scheduler_authority.reconcile_pending()
        assert jobs.get_job(job['id'])['paused_reason'] == 'operator hold'
