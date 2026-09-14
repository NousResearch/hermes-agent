"""Terminal owner receipts recover delivery without replay or double bookkeeping."""
import asyncio
import json
from types import SimpleNamespace

from cron import jobs, scheduler
from gateway import run, session_cron
from gateway.session import SessionStore
from gateway.session_authority import SessionAuthority
from gateway.config import GatewayConfig
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, claim_session_input, settle_session_input


def test_lost_ack_reconciles_frozen_delivery_once(tmp_path, monkeypatch):
    from cron import scheduler_authority, delivery_queue
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='original', schedule='every 1h', deliver='local')
        db = SessionDB(tmp_path / 'state.db')
        runner = SimpleNamespace(_draining=False, session_store=SessionStore(tmp_path / 'sessions', GatewayConfig()), adapters={})
        authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='test', db=db,
                                     epoch=begin_runtime_epoch(db, instance_id='test'))
        monkeypatch.setattr(authority, '_schedule', lambda ref: None)
        runner.session_authority = authority
        async def probe():
            params = {'job_id': job['id'], 'request_id': 'fire', 'extra_prompt': None}
            receipt = await session_cron.operation(authority, 'submit', params)
            row = claim_session_input(db, epoch=authority.epoch, session_id=receipt['session_id'])
            settle_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'], generation=row['generation'], outcome='completed',
                                 result={'result': {'cron_result': [True, 'document', 'answer', None]}, 'usage': {}})
            root = tmp_path / 'cron' / 'admissions'; root.mkdir()
            journal = scheduler_authority.journal_path(job['id'], 'fire')
            journal.write_text(json.dumps({'params': params, 'receipt': None}))
            session_cron.bind_owner(authority)
            try:
                await asyncio.to_thread(scheduler_authority.reconcile_pending)
                assert delivery_queue.get_status('fire')['status'] == 'pending'
                assert jobs.get_job(job['id'])['last_status'] == 'delivery_queued'
                completed = jobs.get_job(job['id'])['repeat']['completed']
                journal.write_text(json.dumps({'params': params, 'receipt': None}))
                await asyncio.to_thread(scheduler_authority.reconcile_pending)
                assert jobs.get_job(job['id'])['repeat']['completed'] == completed
                assert delivery_queue.claim_next()['content'] == 'answer'
                assert delivery_queue.claim_next() is None
                unknown = dict(params, request_id='unknown')
                accepted = await session_cron.operation(authority, 'submit', unknown)
                claim_session_input(db, epoch=authority.epoch, session_id=accepted['session_id'])
                from hermes_state_runtime import recover_session_inputs
                authority.epoch = begin_runtime_epoch(db, instance_id='restart')
                recover_session_inputs(db, epoch=authority.epoch)
                unknown_journal = scheduler_authority.journal_path(job['id'], 'unknown')
                unknown_journal.write_text(json.dumps({'params': unknown, 'receipt': None}))
                await asyncio.to_thread(scheduler_authority.reconcile_pending)
                assert unknown_journal.exists()
                assert jobs.get_job(job['id'])['state'] == 'paused'
                assert delivery_queue.claim_next() is None
            finally:
                session_cron.unbind_owner(authority)
        try:
            asyncio.run(probe())
        finally:
            db.close()


def test_failed_manual_run_exits_nonzero(monkeypatch):
    from hermes_cli import cron
    monkeypatch.setattr(cron, '_cron_api', lambda **kw: {'success': True, 'job': {'executed': True, 'execution_success': False}})
    assert cron._job_action('run', 'job', 'Triggered') == 1
