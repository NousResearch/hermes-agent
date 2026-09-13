"""Completed scheduled attempts survive a jobs.json rollback, not just a process restart."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_FIRE = """
import json
import os
from pathlib import Path
import subprocess
import sys
from cron import jobs, scheduler
from cron.scheduler_provider import InProcessCronScheduler
if sys.argv[1] == 'builtin':
    scheduler.tick(verbose=False, sync=True)
else:
    provider = InProcessCronScheduler()
    for job in jobs.load_jobs():
        if sys.argv[1] != 'worker':
            provider.fire_due(job['id'], force=sys.argv[1] == 'manual')
            continue
        from cron.executions import mark_execution_handoff_pending
        claim = provider.claim_fire(job['id'])
        if claim is None:
            continue
        mark_execution_handoff_pending(claim['execution_id'])
        home = Path(os.environ['HERMES_HOME'])
        payload = home / (claim['execution_id'] + '.json')
        ack = home / (claim['execution_id'] + '.ready')
        payload.write_text(json.dumps({'job': claim, 'profile_home': str(home),
                                       'multiplex_active': False}))
        subprocess.run([sys.executable, '-m', 'cron.scheduler',
                        '--external-worker-file', str(payload), '--ack-file', str(ack)],
                       stdin=subprocess.DEVNULL, check=True, timeout=60)
        assert json.loads(ack.read_text())['pid'] != os.getpid()
"""


def _fire(home, mode):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('HERMES_', '_HERMES_'))
           and not k.endswith(('_API_KEY', '_TOKEN'))}
    env['HERMES_HOME'] = str(home)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run([sys.executable, '-c', _FIRE, mode], env=env,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('mode', ['builtin', 'provider', 'worker'])
def test_completed_occurrence_survives_restart_and_prestamp_rollback(tmp_path, mode):
    from datetime import timedelta
    from hermes_time import now

    home = tmp_path / mode
    cron = home / 'cron'
    cron.mkdir(parents=True)
    effect = home / 'effects.txt'
    (home / 'scripts').mkdir()
    script = home / 'scripts' / 'effect.py'
    script.write_text(f"from pathlib import Path\np = Path({str(effect)!r})\n"
                      "with p.open('a') as f: f.write('effect\\n')\nprint('done')\n")
    slot = (now() - timedelta(minutes=30)).isoformat()
    job = {'id': 'occurrence', 'name': 'occurrence', 'prompt': '',
           'schedule': {'kind': 'interval', 'minutes': 240},
           'next_run_at': slot, 'enabled': True, 'state': 'scheduled',
           'script': str(script), 'no_agent': True, 'deliver': 'local',
           'repeat': {'times': None, 'completed': 0}}
    snapshot = json.dumps({'jobs': [job]})
    store = cron / 'jobs.json'
    store.write_text(snapshot)
    _fire(home, mode)
    assert effect.read_text().splitlines() == ['effect']
    store.write_text(snapshot)  # no last_dispatch stamp, no post-completion fields
    _fire(home, mode)  # a fresh interpreter, same authoritative ledger
    assert effect.read_text().splitlines() == ['effect'], 'completed occurrence executed twice'

    # A different missed slot must remain runnable despite the completed row.
    job['next_run_at'] = (now() - timedelta(minutes=10)).isoformat()
    store.write_text(json.dumps({'jobs': [job]}))
    _fire(home, mode)
    assert len(effect.read_text().splitlines()) == 2
    # Manual force is not a completion of the pending scheduled slot.
    job['next_run_at'] = (now() - timedelta(minutes=5)).isoformat()
    pending = json.dumps({'jobs': [job]})
    store.write_text(pending)
    _fire(home, 'manual')
    store.write_text(pending)
    _fire(home, mode)
    assert len(effect.read_text().splitlines()) == 4


def test_ledger_migration_and_completion_identity(tmp_path, monkeypatch):
    import sqlite3
    from cron import executions, jobs
    from cron.occurrences import completed_occurrence
    from cron.scheduler_provider import InProcessCronScheduler

    db = tmp_path / 'executions.db'
    monkeypatch.setattr(executions, 'EXECUTIONS_FILE', db)
    # Pre-migration schema, not initialized by the implementation under test.
    with sqlite3.connect(db) as conn:
        conn.execute('''CREATE TABLE executions (
            id TEXT PRIMARY KEY, job_id TEXT NOT NULL, source TEXT NOT NULL,
            process_id TEXT NOT NULL, pid INTEGER NOT NULL, process_started_at INTEGER,
            status TEXT NOT NULL, claimed_at TEXT NOT NULL,
            started_at TEXT, finished_at TEXT, error TEXT)''')
        conn.execute("INSERT INTO executions VALUES "
                     "('legacy','job','builtin','old',-1,NULL,'completed',"
                     "'2026-01-01T00:00:00Z',NULL,NULL,NULL)")
    slot = '2026-01-01T00:00:00+00:00'
    job = {'id': 'job'}
    assert not completed_occurrence(job, slot)
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT id, scheduled_instant FROM executions').fetchall() == [('legacy', None)]
    for status in ('failed', 'unknown', 'running', 'claimed'):
        row = executions.create_execution('job', source='control', scheduled_instant=slot)
        with sqlite3.connect(db) as conn:
            conn.execute('UPDATE executions SET status=? WHERE id=?', (status, row['id']))
        assert not completed_occurrence(job, slot)
    row = executions.create_execution('job', source='builtin', scheduled_instant=slot)
    executions.finish_execution(row['id'], success=True)
    executions.create_execution('job', source='manual')  # latest row is not the completed one
    assert completed_occurrence(job, '2026-01-01T08:00:00+0800')
    assert not completed_occurrence(job, '2026-01-01T00:00:01Z')
    assert not completed_occurrence({'id': 'other'}, slot)

    # Provider capture is before next_run_at advances; adoption cannot lose identity.
    with jobs.use_cron_store(tmp_path / 'cron'):
        stored = jobs.create_job(prompt='test', schedule='every 4h')
        rows = jobs.load_jobs()
        rows[0]['next_run_at'] = slot
        jobs.save_jobs(rows)
        claim = InProcessCronScheduler().claim_fire(stored['id'])
        assert claim is not None
        assert claim['_scheduled_instant'] == slot
        assert claim['next_run_at'] != slot
        assert '_scheduled_instant' not in jobs.load_jobs()[0]
        executions.mark_execution_handoff_pending(claim['execution_id'])
        adopted = executions.adopt_claimed_execution(claim['execution_id'])
        assert adopted is not None
        assert adopted['scheduled_instant'] == slot

        # A runnable legacy wall-clock value cannot establish an exact UTC identity.
        from datetime import timedelta
        from hermes_time import now
        naive = jobs.create_job(prompt='legacy', schedule='every 4h')
        rows = jobs.load_jobs()
        for item in rows:
            if item['id'] == naive['id']:
                item['next_run_at'] = (now() - timedelta(minutes=10)).replace(tzinfo=None).isoformat()
        jobs.save_jobs(rows)
        due = next(item for item in jobs.get_due_jobs() if item['id'] == naive['id'])
        assert due['_scheduled_instant'] is None
