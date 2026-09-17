"""Salt R4 J1/J3 composed probes, ported as regression cells."""
import json, subprocess, sys, os
import pytest
from cron import executions, scheduler, scheduler_delivery, delivery_queue, jobs, bot_chat_delivery
from tests.cron.test_mixed_delivery_retention import setup
from tests.cron.test_r4_salt_journal_edges import reject_manifest, repaired, fast_gateway
from tests.cron.test_r4_salt_recovery_probes import restart


def test_finish_alone_overrides_correct_queued_input(setup,monkeypatch):
    home,cfg,run,sent=setup
    reject_manifest();fast_gateway(monkeypatch)
    original=scheduler.finish_execution
    seen=[]
    def finish(eid,**kwargs):
        kwargs['delivery_outcome']='queued'
        result=original(eid,**kwargs)
        seen.append({'input':'queued','result':result})
        return result
    monkeypatch.setattr(scheduler,'finish_execution',finish)
    job,row=run(True)
    print('FINISH_ALONE',json.dumps({'calls':seen,'journal':executions.journaled_manifests(),'native_sends':len(sent)}))
    assert row['delivery_outcome']=='queued'


def test_interrupted_scheduler_journal_row_pruned_by_next_completion(setup,monkeypatch):
    home,cfg,run,sent=setup
    reject_manifest()
    original=executions.journal_manifest
    def journal_then_shutdown(eid,manifest):
        original(eid,manifest)
        assert scheduler.mark_running_jobs_interrupted('probe shutdown')
    monkeypatch.setattr(executions,'journal_manifest',journal_then_shutdown)
    job,row=run(False)
    assert row and row['delivery_outcome']=='queued'  # fixed contract: pre-send intent keeps the interrupted run queued
    assert executions.journaled_manifests().get(row['id'])
    assert len(sent)==1
    monkeypatch.setattr(executions,'MAX_TERMINAL_EXECUTIONS',1)
    another=executions.create_execution('retention-pressure',source='direct')
    executions.finish_execution(another['id'],success=True,delivery_outcome='not_configured')
    snapshot={'before':row,'after_retention':executions.get_execution(row['id']),'journal':executions.journaled_manifests(),'job':jobs.get_job(job['id']),'native_sends':len(sent)}
    repaired();cfg(True);bot_chat_delivery.drain();restart(home)
    snapshot['after_restart']=executions.get_execution(row['id']);snapshot['receipt']=bot_chat_delivery._records(bot_chat_delivery._root())[0][1]
    print('COMPOSED_SHUTDOWN_PRUNE',json.dumps(snapshot))
    assert snapshot['after_retention'] is not None


def test_normal_wait_repaired_manifest_still_pins_false_outcome(setup,monkeypatch):
    home,cfg,run,sent=setup
    reject_manifest();fast_gateway(monkeypatch)
    job,row=run(True)
    # Pending Bot Chat truly fails; no fabricated terminal receipt.
    from tools import bot_live_delivery as mailbox
    monkeypatch.setattr(mailbox,'find_canonical_owner',lambda h:None)
    monkeypatch.setattr(scheduler_delivery,'_deliver_to_bot_chat',lambda *a,**k:'consumer failure')
    bot_chat_delivery.drain()
    repaired();restart(home)
    after=executions.get_execution(row['id'])
    print('WAIT_BOT_FAILED',json.dumps({'before':row,'after':after,'job':jobs.get_job(job['id']),'receipt':bot_chat_delivery._records(bot_chat_delivery._root())[0][1],'sends':len(sent)}))
    assert after['delivery_outcome']=='unknown'


def test_abandoned_direct_owner_recovery_prunes_outstanding_journal(setup,monkeypatch):
    home,cfg,run,sent=setup
    reject_manifest()
    original=scheduler.finish_execution
    def lose_terminal_write(*a,**kwargs):
        raise OSError('terminal write unavailable until owner dies')
    monkeypatch.setattr(scheduler,'finish_execution',lose_terminal_write)
    job,row=run(False)
    assert row['status']=='running'
    assert executions.journaled_manifests().get(row['id'])
    # Real recovery decisions need a provably absent owner; set the durable pid to a dead pid.
    with executions._transaction() as conn:
        conn.execute('UPDATE executions SET process_id=?,pid=?,process_started_at=? WHERE id=?',('dead-test-owner',99999999,0,row['id']))
    monkeypatch.setattr(executions,'MAX_TERMINAL_EXECUTIONS',0)
    assert executions.recover_interrupted_executions()==1
    print('ABANDONED_JOURNAL',json.dumps({'after':executions.get_execution(row['id']),'journal':executions.journaled_manifests(),'sends':len(sent)}))
    assert executions.get_execution(row['id']) is not None
