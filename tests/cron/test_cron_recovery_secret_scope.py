"""Receipt recovery resolves cron delivery under the owning profile's secret scope.

A multiplexing gateway ticks each profile without a bound secret scope — only the fire body
installs one (``scheduler._fire_secret_scope``) — so the delivery lookup that receipt recovery
runs for a settled admission read ``<PLATFORM>_HOME_CHANNEL`` through ``get_secret`` with no
scope installed, raised ``UnscopedSecretError``, and the admission was deferred on every tick
(field report: one orphaned admission, ~2,880 traceback lines a day, #151918).
"""
from contextlib import asynccontextmanager
import json

from agent import secret_scope
from cron import executions, jobs, scheduler_authority
from hermes_cli import gateway_client


def peer(monkeypatch, call):
    class Peer:
        async def rpc(self, method, **params):
            return call(method, params)

    @asynccontextmanager
    async def connected():
        yield Peer()

    monkeypatch.setattr(gateway_client, 'connect_gateway', connected)


def pending_origin_receipt(tmp_path, monkeypatch):
    """A settled admission whose delivery must resolve the Slack home channel at recovery time."""
    job = jobs.create_job(
        prompt='report', schedule='every 1h', deliver='origin',
        origin={'platform': 'slack', 'chat_id': 'D0HOME', 'thread_id': '1700000000.1',
                'chat_type': 'channel'})
    execution = executions.create_execution(job['id'], source='test')
    executions.mark_execution_running(execution['id'])
    with executions._transaction() as conn:
        conn.execute('UPDATE executions SET process_id=?,pid=? WHERE id=?',
                     ('old-owner', 999999, execution['id']))
    params = {'job_id': job['id'], 'request_id': execution['id'], 'extra_prompt': None}
    journal = scheduler_authority.journal_path(job['id'], execution['id'])
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({'params': params, 'receipt': None}))
    peer(monkeypatch, lambda method, params: {'status': 'terminal', 'job': job,
                                             'result': [True, 'document', 'answer', None]})
    return job, execution, journal


def test_recovery_installs_the_profile_scope_before_resolving_delivery(tmp_path, monkeypatch, caplog):
    token = secret_scope.set_multiplex_context(True)
    try:
        with jobs.use_cron_store(tmp_path / 'cron'):
            monkeypatch.setattr('cron.delivery_queue.DELIVERY_DB', tmp_path / 'deliveries.db')
            _job, execution, journal = pending_origin_receipt(tmp_path, monkeypatch)
            scheduler_authority.reconcile_pending()
            assert 'Cron receipt recovery deferred' not in caplog.text
            assert not journal.exists()
            row = executions.get_execution(execution['id'])
            assert row is not None and row['status'] == 'completed'
    finally:
        secret_scope.reset_multiplex_context(token)


def test_orphaned_admission_is_consumed_not_deferred_forever(tmp_path, monkeypatch, caplog):
    """The field symptom: the job was deleted before its receipt settled, so the admission sat in
    the directory and every tick deferred it again (~2,880 tracebacks a day). Recovery must retire
    it, not re-defer it."""
    token = secret_scope.set_multiplex_context(True)
    try:
        with jobs.use_cron_store(tmp_path / 'cron'):
            monkeypatch.setattr('cron.delivery_queue.DELIVERY_DB', tmp_path / 'deliveries.db')
            job, _execution, journal = pending_origin_receipt(tmp_path, monkeypatch)
            jobs.remove_job(job['id'])
            scheduler_authority.reconcile_pending()
            assert 'Cron receipt recovery deferred' not in caplog.text
            assert not journal.exists()
    finally:
        secret_scope.reset_multiplex_context(token)


def test_recovery_restores_the_outer_scope_it_ran_inside(tmp_path, monkeypatch):
    outer = {'SLACK_HOME_CHANNEL': 'D0OUTER'}
    token = secret_scope.set_multiplex_context(True)
    scope_token = secret_scope.set_secret_scope(outer, profile_home=str(tmp_path))
    try:
        with jobs.use_cron_store(tmp_path / 'cron'):
            monkeypatch.setattr('cron.delivery_queue.DELIVERY_DB', tmp_path / 'deliveries.db')
            _job, _execution, journal = pending_origin_receipt(tmp_path, monkeypatch)
            scheduler_authority.reconcile_pending()
            assert not journal.exists()
            assert secret_scope.current_secret_scope() is outer
    finally:
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.reset_multiplex_context(token)
