"""Salt R4 J2: reconciler that started before journal creation must not settle the placeholder."""
import json, threading
from cron import executions, scheduler, jobs, bot_chat_delivery
from tests.cron.test_mixed_delivery_retention import setup
from tests.cron.test_r4_salt_journal_edges import reject_manifest, fast_gateway


def test_reconciler_started_before_journal_creation_does_not_settle_placeholder(setup,monkeypatch):
    home,cfg,run,sent=setup
    reject_manifest()
    job,row=run(True)
    entered=threading.Event();release=threading.Event();errors=[]
    original=executions._recover_unrecorded_manifests
    def paused():
        outstanding=original()
        if threading.current_thread().name=='early-reconciler':
            assert not outstanding
            entered.set();assert release.wait(15)
        return outstanding
    monkeypatch.setattr(executions,'_recover_unrecorded_manifests',paused)
    def reconcile():
        try:executions.reconcile_delivery_projections()
        except BaseException as e:errors.append(str(e))
    thread=threading.Thread(target=reconcile,name='early-reconciler');thread.start()
    assert entered.wait(15)
    scheduler.drain_delivery_queue({},None)
    assert executions.journaled_manifests().get(row['id'])
    assert executions.get_execution(row['id'])['delivery_outcome']=='queued'
    release.set();thread.join(15)
    assert not thread.is_alive() and not errors
    after=executions.get_execution(row['id'])
    print('STALE_OUTSTANDING',json.dumps({'row':after,'journal':executions.journaled_manifests(),'job':jobs.get_job(job['id']),'native_sends':len(sent)}))
    assert after['delivery_outcome']=='queued'
    assert after['delivery_projection_settled']==0


def test_fast_gateway_without_fault_keeps_bot_queued(setup,monkeypatch):
    home,cfg,run,sent=setup
    fast_gateway(monkeypatch)
    job,row=run(True)
    assert row['delivery_outcome']=='queued'
    assert jobs.get_job(job['id'])['last_delivery_queued']
    cfg(True);bot_chat_delivery.drain()
    assert executions.get_execution(row['id'])['delivery_outcome']=='delivered'
    assert len(sent)==1
    print('HEALTHY_FAST_WAIT',json.dumps(executions.get_execution(row['id'])))
