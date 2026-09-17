"""Salt R6 probes (partial-exception settle, empty-bot outcomes, jobs freshness, K2a/K2b migration authority), ported as regression cells."""
import asyncio
import json
import os
import sqlite3
from pathlib import Path
import pytest
from cron import executions as ex, jobs, scheduler_delivery as sd, bot_chat_delivery as bot, delivery_queue as dq
from tests.cron.test_mixed_delivery_retention import setup
from tests.cron.test_r5_salt_legacy_and_races import store, prior, seed, receipt, snap

@pytest.mark.parametrize('exception_type',[OSError, asyncio.CancelledError, KeyboardInterrupt])
@pytest.mark.parametrize('persist_fault',[False,True])
def test_partial_native_and_bot_then_exception(setup,monkeypatch,exception_type,persist_fault):
    home,cfg,run,sent=setup
    job=jobs.create_job(prompt='probe',schedule='every 1h',deliver='telegram:test,bot-chat')
    row=ex.create_execution(job['id'],source='r6'); job['execution_id']=row['id']
    jobs.bind_delivery_execution(job['id'],row['id'])
    original=sd._deliver_targets
    error=exception_type('r6 post-admission exception')
    def fail_after(*args,**kwargs):
        original(*args,**kwargs)
        assert len(sent)==1 and job['_bot_chat_delivery_receipts']
        raise error
    monkeypatch.setattr(sd,'_deliver_targets',fail_after)
    if persist_fault:
        with ex._transaction() as conn:
            conn.execute("CREATE TRIGGER r6_fault BEFORE UPDATE OF delivery_manifest ON executions BEGIN SELECT RAISE(ABORT,'r6 ledger fault'); END")
    with pytest.raises(exception_type) as caught:
        sd._deliver_result(job,'r6 payload')
    assert caught.value is error
    state=ex.get_execution(row['id'])
    manifest=ex.journaled_manifests()[row['id']] if persist_fault else json.loads(state['delivery_manifest'])
    assert manifest['bot'] and manifest['delivered'] and 'r6 post-admission exception' in manifest['error']
    if persist_fault:
        with ex._transaction() as conn: conn.execute('DROP TRIGGER r6_fault')
        ex._recover_unrecorded_manifests()
    finished=ex.finish_execution(row['id'],success=False,error='original run failure',delivery_outcome='failed')
    assert finished['delivery_outcome']=='queued'
    assert jobs.get_job(job['id'])['last_delivery_queued']
    print('PARTIAL',exception_type.__name__,persist_fault,json.dumps(ex.get_execution(row['id'])))

@pytest.mark.parametrize('error,expected',[(None,'delivered'),('bot discovery failed','failed')])
def test_empty_bot_manifest_native_outcome(setup,monkeypatch,error,expected):
    home,cfg,run,sent=setup
    monkeypatch.setattr(sd,'_deliver_to_bot_chat',lambda *a,**k:error)
    job,row=run(False,deliver='telegram:test,bot-chat')
    assert len(sent)==1
    manifest=json.loads(row['delivery_manifest'])
    assert manifest['bot']=={} and manifest['delivered'] is True
    assert row['delivery_manifest_pending']==0 and row['delivery_outcome']==expected
    print('EMPTY_BOT',expected,json.dumps(row))

@pytest.mark.parametrize('outcome,accepted',[(None,True),('queued',True),('delivered',False),('failed',False),('unknown',False),('suppressed',False),('not_configured',False)])
def test_jobs_freshness_outcomes(store,outcome,accepted):
    eid=seed(store)
    with ex._transaction() as conn:
        conn.execute('UPDATE executions SET delivery_outcome=? WHERE id=?',(outcome,eid))
    values={'last_delivery_queued':{'bot-chat:test':{'status':'queued'}}}
    assert jobs.update_delivery_projection('child-job',eid,values) is accepted
    assert jobs.update_delivery_projection('child-job',eid,{'last_delivery_queued':None}) is True
    assert jobs.get_job('child-job')['last_delivery_queued'] is None
    print('FRESHNESS',outcome,accepted)


def legacy_running(home):
    old=prior(); old.EXECUTIONS_FILE=home/'cron'/'executions.db'
    row=old.create_execution('child-job',source='r6-upgrade'); eid=row['id']
    # Model a surviving process importing upgraded code: retain its ownership.
    with old._transaction() as conn:
        conn.execute('UPDATE executions SET process_id=? WHERE id=?',(ex._PROCESS_ID,eid))
    old.record_delivery_manifest(eid,{'external':True})
    jobs.save_jobs([{'id':'child-job','name':'r6','enabled':False,'schedule':{'kind':'interval','minutes':60},'delivery_execution_id':eid,'last_delivery_queued':{'gateway':{'status':'queued'}}}])
    dq.enqueue(eid,{'id':'child-job'},'evidence'); assert dq.claim_next(); assert dq._finish(eid,error=None)
    manifest=receipt(home,eid); old.journal_manifest(eid,manifest)
    with sqlite3.connect(old.EXECUTIONS_FILE) as conn:
        assert 'delivery_manifest_pending' not in [r[1] for r in conn.execute('PRAGMA table_info(executions)')]
    return eid,manifest

@pytest.mark.parametrize('inaccessible',['directory','entry'])
def test_unreadable_upgrade_must_not_terminalize_placeholder(store,inaccessible):
    eid,manifest=legacy_running(store)
    root=store/'cron'/'manifest_journal'
    target=root if inaccessible=='directory' else root/(eid+'.json')
    original_mode=target.stat().st_mode & 0o777
    target.chmod(0)
    try:
        # Real permission fault: no monkeypatch of journal enumeration or projection.
        after_init=ex.get_execution(eid)
        adopted_while_unreadable=ex._legacy_intent_adopted()
        finished=ex.finish_execution(eid,success=True,delivery_outcome='queued')
        while_unreadable=ex.get_execution(eid)
    finally:
        target.chmod(original_mode)
    ex.reconcile_delivery_projections()
    repaired=ex.get_execution(eid)
    print('UNREADABLE_UPGRADE',inaccessible,json.dumps({'after_init':after_init,'finished':finished,'unreadable':while_unreadable,'restored':repaired,'job':jobs.get_job('child-job')}))
    # Contract (Salt R6 K2a): an unreadable inventory is NOT proof of no outstanding intent. The
    # mechanism here is the migration-completion fence row (absent until adoption succeeds), so
    # every reader holds placeholder-only rows conservatively; the flag itself lands on retry.
    assert not adopted_while_unreadable, 'unreadable adoption must not be recorded as complete'
    assert finished['delivery_outcome']=='queued', 'late finish projected the placeholder while adoption incomplete'
    assert while_unreadable['delivery_projection_settled']==0
    assert repaired['delivery_outcome']=='queued', 'readable journal restoration cannot repair falsely terminal ledger'
    assert '"bot"' in repaired['delivery_manifest'] and jobs.get_job('child-job')['last_delivery_queued']
    assert ex._legacy_intent_adopted(), 'adoption retried and completed once the journal became readable'


def test_failed_adoption_must_retry_before_finish(store):
    eid,manifest=legacy_running(store)
    # Trigger can reference the to-be-added column, as a restored old-schema DB can.
    with sqlite3.connect(ex.EXECUTIONS_FILE) as conn:
        conn.execute("CREATE TRIGGER r6_adoption_fault BEFORE UPDATE ON executions WHEN NEW.delivery_manifest_pending=1 BEGIN SELECT RAISE(ABORT,'r6 adoption fault'); END")
    # Adoption fails (rolled back, logged) but initialization degrades to a fenced state rather
    # than blocking every cron operation; the fence row is absent so readers hold the row.
    ex.get_execution(eid)
    assert not ex._legacy_intent_adopted()
    with sqlite3.connect(ex.EXECUTIONS_FILE) as conn:
        columns=[r[1] for r in conn.execute('PRAGMA table_info(executions)')]
        raw=conn.execute('SELECT delivery_manifest_pending FROM executions WHERE id=?',(eid,)).fetchone()[0]
        conn.execute('DROP TRIGGER r6_adoption_fault')
    # The first retry is a legitimate late worker finish, not a reconciler.
    finished=ex.finish_execution(eid,success=True,delivery_outcome='queued')
    after=ex.get_execution(eid)
    print('ADOPTION_RETRY',json.dumps({'pending_column_survives': 'delivery_manifest_pending' in columns,'raw_pending':raw,'finished':finished,'after':after,'job':jobs.get_job('child-job')}))
    assert after['delivery_outcome']=='queued', 'schema ALTER survived adoption rollback; next finish projects placeholder before replay'
    assert ex._legacy_intent_adopted(), 'adoption retried on the next initialization'
    assert '"bot"' in after['delivery_manifest'], 'replay landed the real manifest after adoption fenced the row'
