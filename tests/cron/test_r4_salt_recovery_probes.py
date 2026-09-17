"""Salt R3 (B1-B3) recovery probes, ported verbatim as regression cells. Real stores/router; network and ledger edges injected."""
import json, os, subprocess, sys, threading
from pathlib import Path
import pytest
from cron import executions, scheduler, bot_chat_delivery, jobs, delivery_queue
from tools import bot_live_delivery as mailbox
from tests.cron.test_mixed_delivery_retention import setup

ROOT=Path(__file__).resolve().parent

def restart(home, *, fault=False, count=1):
    REPO=next(p for p in ROOT.parents if (p/'cron'/'executions.py').exists())
    env={k:v for k,v in os.environ.items() if k in ('PATH','LANG','LC_ALL','PYTHONDONTWRITEBYTECODE')}
    env['PYTHONPATH']=str(REPO)
    env.update(HOME=str(home), HERMES_HOME=str(home), HERMES_MANAGED_DIR=str(home/'managed'), TMPDIR=str(home))
    cmd=[sys.executable,str(ROOT/'_r4_restart.py'),str(int(fault)),str(count)]
    p=subprocess.run(cmd,env=env,cwd=str(REPO),capture_output=True,text=True,timeout=30)
    print('RESTART',p.returncode,p.stdout,p.stderr)
    assert p.returncode==0
    return p

def fail_bot_store(monkeypatch):
    orig=executions._store_manifest
    def fail(conn,eid,manifest):
        if 'bot' in manifest: raise OSError('injected persistent SQLite manifest write failure')
        return orig(conn,eid,manifest)
    monkeypatch.setattr(executions,'_store_manifest',fail)
    return orig

@pytest.mark.parametrize('external',[False,True])
def test_live_mailbox_recorder_fault_recovers(setup,monkeypatch,external):
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_state import SessionDB
    home,cfg,run,sent=setup
    db=SessionDB(db_path=home/'state.db');db.create_session(session_id='live-bot',source='tui');db.set_session_title('bot','old bot');db.set_session_title('live-bot','Bot Chat')
    lease,refusal=try_acquire_active_session(session_id='live-bot',surface='tui',config={},registry_home=home,metadata={'bot_live_delivery_consumer':True,'live_session_id':'salt-live'})
    assert refusal is None
    try:
        owner=mailbox.find_canonical_live_owner(home);assert owner
        orig=executions.record_delivery_manifest
        def fail(eid,manifest):
            if 'bot' in manifest: raise OSError('injected recorder fault')
            return orig(eid,manifest)
        monkeypatch.setattr(executions,'record_delivery_manifest',fail)
        job,before=run(external)
        if external: scheduler.drain_delivery_queue({},None)
        records=list((home/'runtime/bot_live_delivery').glob('*.json'));assert len(records)==1
        key=records[0].stem
        mid=executions.get_execution(before['id'])
        print('LIVE_MID',external,json.dumps(mid), 'pending',bot_chat_delivery.read_pending(key))
        assert len(sent)==1
        claimed=mailbox.claim_pending_delivery(home,owner);assert claimed
        mailbox.complete_delivery(home,key,status='settled',reply='done')
        monkeypatch.setattr(executions,'record_delivery_manifest',orig)
        restart(home)
        after=executions.get_execution(before['id']);saved=jobs.get_job(job['id'])
        print('LIVE_FINAL',external,json.dumps(after),'queued',saved.get('last_delivery_queued'))
        assert after['delivery_outcome']=='delivered'
        assert not saved.get('last_delivery_queued')
        assert after['delivery_manifest'] and 'bot' in json.loads(after['delivery_manifest'])
    finally: lease.release();db.close()

@pytest.mark.parametrize('external',[False,True])
def test_persistent_real_manifest_store_fault_across_restart(setup,monkeypatch,external):
    home,cfg,run,sent=setup
    orig=fail_bot_store(monkeypatch)
    job,before=run(external)
    if external: scheduler.drain_delivery_queue({},None)
    assert executions.journaled_manifests().get(before['id'])
    restart(home,fault=True,count=2)
    mid=executions.get_execution(before['id']);saved=jobs.get_job(job['id'])
    print('PERSISTENT_MID',external,json.dumps(mid),'queued',saved.get('last_delivery_queued'))
    assert mid['delivery_outcome']=='queued'
    assert saved.get('last_delivery_queued')
    monkeypatch.setattr(executions,'_store_manifest',orig)
    cfg(True);bot_chat_delivery.drain();restart(home)
    after=executions.get_execution(before['id'])
    assert after['delivery_outcome']=='delivered'
    assert not jobs.get_job(job['id']).get('last_delivery_queued')
    assert len(sent)==1




def test_parked_manifest_execution_is_not_pruned_during_store_fault(setup,monkeypatch):
    home,cfg,run,sent=setup
    fail_bot_store(monkeypatch)
    job,before=run(True);scheduler.drain_delivery_queue({},None)
    assert executions.journaled_manifests().get(before['id'])
    monkeypatch.setattr(executions,'MAX_TERMINAL_EXECUTIONS',0)
    with executions._transaction() as conn:executions._prune_unlocked(conn)
    print('PARKED_PRUNE',executions.get_execution(before['id']))
    assert executions.get_execution(before['id']) is not None
