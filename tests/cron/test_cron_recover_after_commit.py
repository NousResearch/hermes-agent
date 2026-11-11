"""A cron submit that fails after the owner committed the admission is observed, not paused (N61).

The firer's ``submit`` raised after the admission row committed (a lost answer / an owner error
past the commit). ``recover`` found the run, but the firer re-raised anyway: CronExecutionUnknown
paused the job and gave up a run the owner then completed. Real SessionAuthority over SQLite.
"""
import asyncio

from cron import executions, jobs, scheduler
from gateway import session_cron
from hermes_state_runtime import claim_session_input, settle_session_input
from tests.cron.test_cron_recovery import _owner


def test_submit_error_after_commit_observes_the_admitted_run(tmp_path, monkeypatch):
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')
        authority, db = _owner(tmp_path, monkeypatch)
        fire = executions.create_execution(job['id'], source='test')['id']
        real = session_cron.operation

        async def lossy(owner, name, params, actor=None):
            answer = await real(owner, name, params, actor)
            if name != 'submit':
                return answer
            # The owner admitted (and here, ran) the fire; the firer never sees the receipt.
            row = claim_session_input(db, epoch=authority.epoch, session_id=answer['session_id'])
            settle_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'],
                                 generation=row['generation'], outcome='completed',
                                 result={'result': {'cron_result': [True, 'document', 'answer', None]}, 'usage': {}})
            raise RuntimeError('owner answer lost after commit')

        monkeypatch.setattr(session_cron, 'operation', lossy)

        async def probe():
            session_cron.bind_owner(authority)
            try:
                return await asyncio.to_thread(
                    scheduler._run_one_job_body, dict(jobs.get_job(job['id']), execution_id=fire))
            finally:
                session_cron.unbind_owner(authority)
        try:
            asyncio.run(probe())
        finally:
            db.close()
        assert jobs.get_job(job['id'])['state'] != 'paused'
        assert executions.get_execution(fire)['status'] == 'completed'
