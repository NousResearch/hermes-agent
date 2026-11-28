"""A cron job deleted between the tick and the submit fails with main's refusal text.

The in-process owner refuses the submit with the store's ``not_found`` refusal, ``recover`` finds no
admission, and the run's stored failure names the refusal as ``RuntimeStoreError``, as on main, not as
the interface's ``AuthorityRefusal`` class. Real SessionAuthority over SQLite.
"""
import asyncio

from cron import jobs
from cron.scheduler_authority import run_canonical_job
from gateway import session_cron
from tests.cron.test_cron_recovery import _owner


def test_a_job_deleted_before_submit_fails_with_mains_refusal_name(tmp_path, monkeypatch):
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')
        authority, db = _owner(tmp_path, monkeypatch)
        fired = dict(jobs.get_job(job['id']), execution_id='fire-deleted')
        jobs.remove_job(job['id'])

        async def probe():
            session_cron.bind_owner(authority)
            try:
                return await asyncio.to_thread(run_canonical_job, fired, execution_id='fire-deleted')
            finally:
                session_cron.unbind_owner(authority)
        try:
            ok, _doc, _final, error = asyncio.run(probe())
        finally:
            db.close()
        assert ok is False
        assert error == 'RuntimeStoreError: not_found'
