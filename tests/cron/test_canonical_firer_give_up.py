"""An in-gateway firer that gives up on a canonical run records its own attempt as given up."""
from contextlib import asynccontextmanager
import json

from cron import executions, jobs, scheduler, scheduler_authority
from hermes_cli import gateway_client


def test_given_up_fire_leaves_its_attempt_recoverable_not_running(tmp_path, monkeypatch):
    """CronExecutionUnknown pauses the job, but the firer's own ``running`` row must not stay in
    flight until a restart: receipt recovery skips a same-process running row, so the owner's
    later terminal receipt could never settle it (R17 residual)."""
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')
        fire = executions.create_execution(job['id'], source='test')['id']
        journal = scheduler_authority.journal_path(job['id'], fire)

        def run_job(job, **kwargs):
            journal.parent.mkdir(parents=True, exist_ok=True)
            journal.write_text(json.dumps({'params': {'job_id': job['id'], 'request_id': fire,
                                                      'extra_prompt': None}, 'receipt': None}))
            raise scheduler_authority.CronExecutionUnknown('Cron execution unverified: owner lost')
        monkeypatch.setattr(scheduler, 'run_job', run_job)
        assert scheduler._run_one_job_body(dict(jobs.get_job(job['id']), execution_id=fire)) is False
        assert jobs.get_job(job['id'])['state'] == 'paused'
        given_up = executions.get_execution(fire)
        assert given_up['status'] == 'unknown' and 'owner lost' in given_up['error']

        class Peer:
            async def rpc(self, method, **params):
                return {'status': 'terminal', 'job': job, 'result': [True, 'document', 'answer', None]}

        @asynccontextmanager
        async def connected():
            yield Peer()
        monkeypatch.setattr(gateway_client, 'connect_gateway', connected)
        scheduler_authority.reconcile_pending()
        assert executions.get_execution(fire)['status'] == 'completed'
        assert not journal.exists()
