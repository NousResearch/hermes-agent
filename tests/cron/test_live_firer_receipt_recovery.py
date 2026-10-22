"""Receipt reconciliation keeps its journal while a live foreign firer still owns the ledger row."""
from contextlib import asynccontextmanager
import json
import os
import subprocess
import sys

from cron import executions, jobs, scheduler_authority


def test_live_firer_journal_survives_until_its_execution_can_settle(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')
        code = ('import sys; from cron.executions import create_execution, mark_execution_running; '
                f'row=create_execution({job["id"]!r},source="receipt-fixture"); '
                'mark_execution_running(row["id"]); print(row["id"],flush=True); sys.stdin.read()')
        child = subprocess.Popen([sys.executable, '-c', code], env=dict(os.environ),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            fire = child.stdout.readline().strip()
            assert fire and child.poll() is None
            assert executions.get_execution(fire)['status'] == 'running'
            params = {'job_id': job['id'], 'request_id': fire, 'extra_prompt': None}
            journal = scheduler_authority.journal_path(job['id'], fire)
            journal.parent.mkdir(parents=True)
            journal.write_text(json.dumps({'params': params, 'receipt': None}), encoding='utf-8')
            class Peer:
                async def rpc(self, method, **kwargs):
                    assert method == 'cron.recover' and kwargs == params
                    return {'status': 'terminal', 'job': job, 'result': [True, 'report', 'answer', None]}
            @asynccontextmanager
            async def connect():
                yield Peer()
            monkeypatch.setattr('hermes_cli.gateway_client.connect_gateway', connect)
            for _ in range(2):
                scheduler_authority.reconcile_pending()
            assert journal.exists(), 'the live firer left running execution without its recovery receipt'
            assert executions.get_execution(fire)['status'] == 'running'
            # Retaining the journal must not replay the receipt's side effects on every tick.
            assert not list(jobs._job_output_dir(job['id']).glob('*.md'))
            assert jobs.get_job(job['id'])['last_run_at'] is None
            child.stdin.close()
            child.wait(timeout=10)
            scheduler_authority.reconcile_pending()
            assert not journal.exists()
            assert executions.get_execution(fire)['status'] == 'completed'
            assert jobs.get_job(job['id'])['repeat']['completed'] == 1
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
            if not child.stdin.closed:
                child.stdin.close()
            child.stdout.close()
