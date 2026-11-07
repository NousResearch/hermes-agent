"""A recurring canonical cron job's completion tombstones stay bounded: an id is retired once its
admission journal is gone (nothing can replay it), never while recovery could still replay it."""
from cron import jobs, scheduler_authority


def _fire(job_id, eid, *, retire_journal=True):
    journal = scheduler_authority.journal_path(job_id, eid)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text('{}', encoding='utf-8')  # run_canonical_job writes it before admission
    assert jobs.mark_job_run(job_id, True, execution_id=eid) is True
    if retire_journal:
        journal.unlink()  # finish_completed_run's tail


def test_completion_ids_retire_with_their_journal(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    with jobs.use_cron_store(tmp_path / 'cron'):
        job_id = jobs.create_job(prompt='report', schedule='every 1m', deliver='local')['id']
        _fire(job_id, 'pending', retire_journal=False)  # a tail that crashed before unlinking
        for n in range(64):
            _fire(job_id, f'run-{n}')
        assert jobs.get_job(job_id)['canonical_completions'] == ['pending', 'run-63']
        completed = jobs.get_job(job_id)['repeat']['completed']
        # The still-journaled completion replayed by recovery is a no-op, not a second run.
        assert jobs.mark_job_run(job_id, True, execution_id='pending') is True
        assert jobs.get_job(job_id)['repeat']['completed'] == completed
