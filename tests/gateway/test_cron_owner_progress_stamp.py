"""An owner-side canonical cron run proves progress on the FIRER's ledger row.

The owner executes the job, but the execution row belongs to the firer (an external worker), and
``touch_execution_progress`` is fenced to the row's owner. Before this fix the owner's job dict had
no execution id at all (its stamper returned early) and the firer never stamped, so a long, healthy
run was swept ``unknown`` once the derived stale bound passed."""
import asyncio
import threading
import time
from types import SimpleNamespace

import cron.executions as executions_mod


def test_owner_progress_reaches_the_firers_ledger_row(tmp_path, monkeypatch):
    from cron import jobs, scheduler, scheduler_liveness
    from cron.executions import _transaction, create_execution, mark_execution_running
    from cron.scheduler_authority import run_canonical_job
    from gateway import run, session_cron
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_local_recovery import local_adapter_map
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input, settle_session_input

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(executions_mod, 'EXECUTIONS_FILE', tmp_path / 'cron' / 'executions.db')
    monkeypatch.setattr(jobs, 'get_job', lambda jid: {'id': jid, 'prompt': 'p', 'model': 'm'})
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    # The owner is not the row's owner: its own stamp never lands on the firer's row.
    monkeypatch.setattr(scheduler_liveness, 'touch_execution_progress', lambda execution_id: False)
    execution = create_execution('job', source='cron')
    mark_execution_running(execution['id'])
    db = SessionDB(tmp_path / 'state.db')
    runner = SimpleNamespace(_draining=False, session_store=SessionStore(tmp_path / 'sessions', GatewayConfig()),
                             adapters={})
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='test', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='test'))
    runner.session_authority = authority
    admitted = threading.Event()
    monkeypatch.setattr(authority, '_schedule', lambda ref: admitted.set())

    def run_job(job, *, extra_prompt, execution_id, cancel_event):
        stamper = scheduler_liveness.ExecutionProgressStamper(
            str(job.get('execution_id') or ''), 'job', idle_seconds=lambda: 0.0, every_seconds=0.001)
        time.sleep(0.01)
        stamper.tick()
        time.sleep(0.6)  # the firer polls status while the agent works
        return (True, 'doc', 'answer', None)
    monkeypatch.setattr(scheduler, 'run_job', run_job)

    async def probe():
        session_cron.bind_owner(authority)
        try:
            firer = asyncio.create_task(asyncio.to_thread(run_canonical_job, {'id': 'job'},
                                                          execution_id=execution['id']))
            assert await asyncio.to_thread(admitted.wait, 10), firer
            sid = next(iter(authority.sessions))
            row = claim_session_input(db, epoch=authority.epoch, session_id=sid)
            policy = local_adapter_map(authority)[Platform.LOCAL].policies[sid]
            await session_cron.execute(authority, SessionRef(authority.profile_id, sid), row, policy)
            settle_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'],
                                 generation=row['generation'], outcome='completed',
                                 result={'result': {'cron_result': [True, 'doc', 'answer', None]}, 'usage': {}})
            assert await asyncio.wait_for(firer, 10) == (True, 'doc', 'answer', None)
        finally:
            session_cron.unbind_owner(authority)

    try:
        asyncio.run(probe())
    finally:
        db.close()
    with _transaction() as conn:
        stamped = conn.execute('SELECT progress_at FROM executions WHERE id=?', (execution['id'],)).fetchone()[0]
    assert stamped is not None, "the firer's row never saw the owner's progress"
