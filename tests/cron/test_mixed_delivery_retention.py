"""Independent product-path falsification; vendor edges only are fake."""
import json
from pathlib import Path
import pytest
from cron import executions, incidents, jobs, scheduler, delivery_queue, bot_chat_delivery
from gateway.config import GatewayConfig, Platform, PlatformConfig
from hermes_state import SessionDB
from hermes_cli.active_sessions import try_acquire_active_session

@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    monkeypatch.setenv('HERMES_MANAGED_DIR',str(tmp_path/'managed'))
    def cfg(muted=False):
        (tmp_path/'config.yaml').write_text(json.dumps({'cron':{'wrap_response':False,'preflight':False},'display':{'suppress_warning_notifications':muted}}))
    cfg()
    config=GatewayConfig(); config.platforms[Platform.TELEGRAM]=PlatformConfig(enabled=True)
    monkeypatch.setattr('gateway.config.load_gateway_config',lambda:config)
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts'/'fail.sh').write_text("#!/bin/sh\nprintf 'preserved shell stdout\\n'\nprintf 'preserved stderr\\n' >&2\nexit 7\n")
    db=SessionDB(db_path=tmp_path/'state.db'); db.create_session(session_id='bot',source='cli'); db.set_session_title('bot','Bot Chat')
    lease, refusal=try_acquire_active_session(session_id='bot',surface='cli',config={},registry_home=tmp_path)
    assert refusal is None
    sent=[]
    async def send(platform,pconfig,chat_id,text,**kwargs):
        sent.append(text)
        return {'success':True,'message_id':'recorded'}
    monkeypatch.setattr('tools.send_message_tool._send_to_platform',send)
    def run(external=False, deliver='bot-chat,telegram:test'):
        job=jobs.create_job(prompt=None,schedule='every 1h',script='fail.sh',no_agent=True,deliver=deliver)
        if external:
            ex=executions.create_execution(job['id'],source='test-external'); job['execution_id']=ex['id']
            monkeypatch.setenv('_HERMES_CRON_EXTERNAL_WORKER',ex['id'])
            wait=delivery_queue.enqueue_and_wait
            monkeypatch.setattr(delivery_queue,'enqueue_and_wait',lambda eid,j,c,for_failure=False: wait(eid,j,c,for_failure=for_failure,timeout=0))
        scheduler.run_one_job(job)
        return job, executions.latest_execution(job['id'])
    yield tmp_path,cfg,run,sent
    lease.release(); db.close()

@pytest.mark.parametrize('external',[False,True])
@pytest.mark.parametrize('native_fails',[False,True])
def test_real_mixed_router_preserves_pending_and_terminal(setup,monkeypatch,external,native_fails):
    home,cfg,run,sent=setup
    if native_fails:
        async def fail(*a,**k): return {'success':False,'error':'native rejected'}
        monkeypatch.setattr('tools.send_message_tool._send_to_platform',fail)
    job,before=run(external)
    if external:
        assert before['delivery_outcome']=='queued'
        assert scheduler.drain_delivery_queue({},None)==1
    middle=executions.get_execution(before['id']); saved=jobs.get_job(job['id'])
    print('MIXED',external,native_fails,'MID',json.dumps({'execution':middle,'queued':saved.get('last_delivery_queued')},default=str))
    # The bot has NOT been attempted; native failure must not erase its pending state.
    assert middle['delivery_outcome']=='queued'
    assert saved.get('last_delivery_queued')
    cfg(True); bot_chat_delivery.drain(); bot_chat_delivery.drain()
    after=executions.get_execution(before['id'])
    assert after['delivery_outcome']==('failed' if native_fails else 'delivered')
    assert after['error']==before['error'] and after['finished_at']==before['finished_at']
    assert not jobs.get_job(job['id']).get('last_delivery_queued')


def test_execution_retention_does_not_erase_pending_projection(setup,monkeypatch):
    home,cfg,run,sent=setup
    job,before=run(True,deliver='telegram:test')
    # Use the production retention implementation with a small deterministic limit.
    monkeypatch.setattr(executions,'MAX_TERMINAL_EXECUTIONS',1)
    newer=executions.create_execution('unrelated-job',source='retention-pressure')
    executions.finish_execution(newer['id'],success=True,delivery_outcome='not_configured')
    print('PRUNE',before['id'],executions.get_execution(before['id']))
    cfg(True); scheduler.drain_delivery_queue({},None)
    assert delivery_queue.get_status(before['id'])['status']=='suppressed'
    print('PRUNE JOB',json.dumps(jobs.get_job(job['id']),default=str))
    assert not jobs.get_job(job['id']).get('last_delivery_queued')


@pytest.mark.parametrize("drop_settlement_column", [False, True])
def test_projection_write_failure_pins_receipt_until_replay_then_prunes(setup, monkeypatch, drop_settlement_column):
    import sqlite3
    home, cfg, run, sent = setup
    job, before = run(True, deliver="telegram:test")
    if drop_settlement_column:
        with sqlite3.connect(home / "cron" / "executions.db") as conn:
            conn.execute("ALTER TABLE executions DROP COLUMN delivery_projection_settled")
    original = jobs.update_delivery_projection
    def fail(*args, **kwargs):
        raise OSError("fixture jobs projection write failure")
    monkeypatch.setattr(jobs, "update_delivery_projection", fail)
    cfg(True)
    scheduler.drain_delivery_queue({}, None)
    assert executions.get_execution(before["id"])["delivery_outcome"] == "suppressed"
    assert not executions.get_execution(before["id"])["delivery_projection_settled"]
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    with executions._transaction() as conn:
        executions._prune_unlocked(conn)
    assert executions.get_execution(before["id"]) is not None
    assert jobs.get_job(job["id"])["last_delivery_queued"]
    monkeypatch.setattr(jobs, "update_delivery_projection", original)
    executions.reconcile_delivery_projections()
    assert not jobs.get_job(job["id"])["last_delivery_queued"]
    assert executions.get_execution(before["id"])["delivery_projection_settled"]
    with executions._transaction() as conn:
        executions._prune_unlocked(conn)
    assert executions.get_execution(before["id"]) is None
    assert not sent  # never retry delivery to repair bookkeeping

