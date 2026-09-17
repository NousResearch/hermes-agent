"""Salt R4 J1-J4 journal-boundary probes, ported as regression cells (real stores; scheduling/fault edges only)."""
import json, os, subprocess, sys, threading
from pathlib import Path
import pytest
from cron import executions, scheduler, scheduler_delivery, delivery_queue, jobs, bot_chat_delivery
from tests.cron.test_mixed_delivery_retention import setup
from tests.cron.test_r4_salt_recovery_probes import restart


def reject_manifest():
    with executions._transaction() as conn:
        conn.execute('''CREATE TRIGGER salt_r4_reject BEFORE UPDATE OF delivery_manifest ON executions
        WHEN NEW.delivery_manifest LIKE '%"bot"%'
        BEGIN SELECT RAISE(ABORT, 'persistent ledger fault'); END''')


def repaired():
    with executions._transaction() as conn:
        conn.execute('DROP TRIGGER salt_r4_reject')


def fast_gateway(monkeypatch):
    wait=delivery_queue.enqueue_and_wait
    def synchronous(eid,job,content,*,for_failure=False,**kwargs):
        delivery_queue.enqueue(eid,job,content,for_failure=for_failure)
        scheduler.drain_delivery_queue({},None)
        return wait(eid,job,content,for_failure=for_failure,timeout=0)
    monkeypatch.setattr(delivery_queue,'enqueue_and_wait',synchronous)


def test_worker_wait_with_outstanding_journal_keeps_queued(setup,monkeypatch):
    home,cfg,run,sent=setup
    reject_manifest();fast_gateway(monkeypatch)
    job,row=run(True)
    snapshot={'execution':row,'saved_job':jobs.get_job(job['id']),'journal':executions.journaled_manifests(),'queue':delivery_queue.get_status(row['id']),'native_sends':len(sent)}
    print('WAIT_BEFORE_REPAIR',json.dumps(snapshot))
    repaired();cfg(True);bot_chat_delivery.drain();restart(home)
    print('WAIT_AFTER_REPAIR',json.dumps(executions.get_execution(row['id'])))
    assert row['delivery_outcome']=='queued'
    assert snapshot['saved_job']['last_delivery_queued']


def test_finish_projection_cannot_override_queued_with_placeholder(setup,monkeypatch):
    home,cfg,run,sent=setup
    reject_manifest()
    original=executions._delivery_projection
    observed=[]
    def recording(record):
        result=original(record)
        observed.append((record['id'],result))
        return result
    monkeypatch.setattr(executions,'_delivery_projection',recording)
    fast_gateway(monkeypatch)
    job,row=run(True)
    print('FINISH_PROJECTIONS',json.dumps({'observed':observed,'outcome':row['delivery_outcome'],'outstanding':list(executions.journaled_manifests())}))
    # This row is terminal but falsely delivered while a real Bot Chat receipt remains queued.
    assert not any(result and result[0]=='delivered' for eid,result in observed)


@pytest.mark.parametrize('external',[False,True])
def test_unwritable_journal_after_real_send_is_not_transport_failure(setup,monkeypatch,external):
    home,cfg,run,sent=setup
    reject_manifest()
    root=home/'cron'/'manifest_journal';root.mkdir();root.chmod(0o500)
    try:
        job,row=run(external)
        if external:scheduler.drain_delivery_queue({},None)
        snapshot={'execution':executions.get_execution(row['id']),'job':jobs.get_job(job['id']),'queue':delivery_queue.get_status(row['id']) if external else None,'native_sends':len(sent),'receipts':[r for p,r in bot_chat_delivery._records(bot_chat_delivery._root())]}
        print('UNWRITABLE_JOURNAL',json.dumps(snapshot))
        assert len(sent)==1
        if external:assert snapshot['queue']['status']!='failed'
        else:assert not snapshot['job'].get('last_delivery_error')
    finally:root.chmod(0o700)


def test_retention_after_interrupted_direct_run_preserves_journal(setup,monkeypatch):
    home,cfg,run,sent=setup
    reject_manifest()
    ex=executions.create_execution('interrupted',source='direct')
    executions.mark_execution_running(ex['id'])
    manifest={'bot':{'bot-chat:':{'home':str(home),'delivery_id':'receipt','status':'queued'}}}
    executions.journal_manifest(ex['id'],manifest)
    # _finish_interrupted_run calls finish_execution without delivery_outcome.
    monkeypatch.setattr(executions,'MAX_TERMINAL_EXECUTIONS',0)
    scheduler._finish_interrupted_run({'id':'interrupted'},ex['id'],None)
    print('INTERRUPTED_RETENTION',json.dumps({'row':executions.get_execution(ex['id']),'journal':executions.journaled_manifests()}))
    assert executions.get_execution(ex['id']) is not None


def test_journal_without_row_is_retained_not_resurrected(setup):
    home,cfg,run,sent=setup
    manifest={'bot':{'bot-chat:':{'home':str(home),'delivery_id':'missing','status':'queued'}}}
    executions.journal_manifest('never-created',manifest)
    restart(home,count=2)
    assert executions.get_execution('never-created') is None
    assert executions.journaled_manifests()['never-created']==manifest
    print('ORPHAN_RETAINED',json.dumps(executions.journaled_manifests()))


@pytest.mark.parametrize('terminal',['settled','ambiguous','suppressed'])
def test_two_reconcilers_terminal_receipt_replay(setup,monkeypatch,terminal):
    home,cfg,run,sent=setup
    reject_manifest();job,row=run(True);scheduler.drain_delivery_queue({},None)
    from tools import bot_live_delivery as mailbox
    monkeypatch.setattr(mailbox,'find_canonical_owner',lambda h:None)
    if terminal=='suppressed':cfg(True)
    else:monkeypatch.setattr(scheduler_delivery,'_deliver_to_bot_chat',lambda *a,**k: None if terminal=='settled' else 'consumer failed')
    bot_chat_delivery.drain()
    records=bot_chat_delivery._records(bot_chat_delivery._root())
    assert records[0][1]['status']==terminal
    repaired()
    REPO=next(p for p in Path(__file__).resolve().parents if (p/'cron'/'executions.py').exists())
    env={k:v for k,v in os.environ.items() if k in ('PATH','LANG','LC_ALL','PYTHONDONTWRITEBYTECODE')}
    env.update(HOME=str(home),HERMES_HOME=str(home),HERMES_MANAGED_DIR=str(home/'managed'),TMPDIR=str(home),PYTHONPATH=str(REPO))
    script=Path(__file__).parent/'_r4_reconcile_child.py'
    children=[subprocess.Popen([sys.executable,str(script)],env=env,cwd=str(next(p for p in Path(__file__).resolve().parents if (p/'cron'/'executions.py').exists())),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
    for p in children:
        out,err=p.communicate(timeout=30);print('CONCURRENT_REPLAY',p.returncode,out,err);assert p.returncode==0
    after=executions.get_execution(row['id'])
    assert after['delivery_outcome']==('unknown' if terminal=='ambiguous' else 'delivered')
    assert after['delivery_projection_settled']==1
    assert not executions.journaled_manifests()
    assert not jobs.get_job(job['id']).get('last_delivery_queued')
    assert len(sent)==1
    print('CONCURRENT_FINAL',json.dumps(after))
