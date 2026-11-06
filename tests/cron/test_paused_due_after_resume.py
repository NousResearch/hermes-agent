"""A recurring job paused across its slot is not due, and resume keeps that past slot."""
from cron import jobs


def test_paused_job_is_not_due_until_resumed(tmp_path, monkeypatch):
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 10m', deliver='local')
        past = '2026-10-18T18:17:16+09:00'
        jobs.update_job(job['id'], {'next_run_at': past})
        jobs.pause_job(job['id'], reason='Canonical cron unknown; no automatic re-execution')
        assert jobs.get_due_jobs() == []
        resumed = jobs.resume_job(job['id'])
        assert resumed['state'] == 'scheduled'
        assert resumed['next_run_at'] == past
        due = jobs.get_due_jobs()
        assert [j['id'] for j in due] == [job['id']]
