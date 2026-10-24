"""Monitor preparation reads and persists only the owning profile (#140222)."""
import asyncio
import json
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_same_fire_in_two_profiles_keeps_jobs_and_results_separate(tmp_path, monkeypatch):
    from cron import monitor
    from gateway import session_cron
    from gateway.session_authority import SessionAuthority
    from gateway.session_runtime_workers import mutation_tasks
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    homes = [tmp_path / 'alpha', tmp_path / 'beta']
    owners = []
    for home in homes:
        (home / 'cron').mkdir(parents=True)
        job = {'id': 'shared-job', 'prompt': home.name + '-private-prompt',
               'monitor_url': 'https://example.invalid/' + home.name,
               'monitor_state': {'last_output_hash': monitor.hash_monitor_output(home.name)}}
        (home / 'cron' / 'jobs.json').write_text(json.dumps({'jobs': [job]}))
        db = SessionDB(home / 'state.db')
        runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=True))
        authority = SessionAuthority(runner, profile_id=str(home), instance_id=home.name, db=db,
                                     epoch=begin_runtime_epoch(db, instance_id=home.name))
        runner.session_authority = authority
        owners.append(authority)
    samples = []
    def sample(job):
        name = get_hermes_home().name
        samples.append((name, job['monitor_url']))
        return True, name
    monkeypatch.setattr(monitor, '_run_monitor_source', sample)
    params = {'job_id': 'shared-job', 'request_id': 'same-fire', 'extra_prompt': None}
    try:
        await asyncio.gather(*(session_cron.operation(a, 'submit_v2', params) for a in owners))
        await asyncio.gather(*(task for a in owners for task in mutation_tasks(a)))
        for a, home in zip(owners, homes):
            result = await session_cron.operation(a, 'recover_v2', params)
            assert result['status'] == 'terminal' and result['result'][2] == '[SILENT]'
            assert result['job']['prompt'] == home.name + '-private-prompt'
            assert a.db._conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 0
            stored = ' '.join(r[0] for r in a.db._conn.execute(
                "SELECT value FROM state_meta WHERE key GLOB 'gateway.cron_monitor.v1:*'"))
            other = 'beta' if home.name == 'alpha' else 'alpha'
            assert home.name + '-private-prompt' in stored
            assert other + '-private-prompt' not in stored
        assert sorted(samples) == [(h.name, 'https://example.invalid/' + h.name) for h in homes]
    finally:
        for a in owners:
            a.db.close()
