"""Salt R5 K1 intent-lifecycle probes, ported as regression cells (production stores/scheduler; fault edges only)."""
import json, os, subprocess, sys
from pathlib import Path
import pytest
from cron import executions, scheduler, scheduler_delivery, jobs, bot_chat_delivery
from tests.cron.test_mixed_delivery_retention import setup
from tests.cron.test_r4_salt_recovery_probes import restart


def snapshot(home, job, row, sent):
    return {'row':executions.get_execution(row['id']), 'job':jobs.get_job(job['id']),
            'receipts':bot_chat_delivery._records(bot_chat_delivery._root()),
            'journal':executions.journaled_manifests(),'native_sends':len(sent)}


def test_all_suppressed_real_receipt_clears_intent(setup):
    home,cfg,run,sent=setup
    cfg(True)
    job,row=run(False,deliver='bot-chat')
    result=snapshot(home,job,row,sent)
    print('ALL_SUPPRESSED',json.dumps(result,default=str))
    assert row['delivery_manifest_pending']==0
    assert row['delivery_outcome']=='suppressed'
    assert not sent


def test_rejected_before_receipt_clears_intent(setup,monkeypatch):
    home,cfg,run,sent=setup
    from tools import bot_live_delivery
    def fail(*a,**k): raise OSError('discovery unavailable before admission')
    monkeypatch.setattr(bot_live_delivery,'read_delivery_result',fail)
    job,row=run(False,deliver='bot-chat')
    print('PRE_RECEIPT_REJECT',json.dumps(snapshot(home,job,row,sent),default=str))
    assert row['delivery_manifest_pending']==0
    assert row['delivery_outcome']=='failed'
    assert not bot_chat_delivery._records(bot_chat_delivery._root())


def test_presend_ledger_refusal_sends_nothing(setup,monkeypatch):
    home,cfg,run,sent=setup
    with executions._transaction() as conn:
        conn.execute("CREATE TRIGGER reject_intent BEFORE UPDATE OF delivery_manifest_pending ON executions WHEN NEW.delivery_manifest_pending=1 BEGIN SELECT RAISE(ABORT,'intent unavailable'); END")
    job,row=run(False)
    print('PRE_SEND_REFUSAL',json.dumps(snapshot(home,job,row,sent),default=str))
    assert not sent and not bot_chat_delivery._records(bot_chat_delivery._root())
    assert row['delivery_outcome']=='failed' and row['delivery_manifest_pending']==0


def test_exception_before_first_send_does_not_pin_forever(setup,monkeypatch):
    home,cfg,run,sent=setup
    def fail(*a,**k): raise OSError('transport discovery failed before any send')
    monkeypatch.setattr(scheduler_delivery,'_resolve_target_transport',fail)
    job,row=run(False,deliver='telegram:test,bot-chat')
    restart(home,count=2)
    result=snapshot(home,job,row,sent)
    print('NO_RECEIPT_EXCEPTION',json.dumps(result,default=str))
    assert not sent and not result['receipts'] and not result['journal']
    assert result['row']['delivery_manifest_pending']==0, 'Known pre-send exception has no recovery source yet remains permanently pending'


def test_transient_release_fault_recovers_no_receipt(setup,monkeypatch):
    home,cfg,run,sent=setup
    from tools import bot_live_delivery
    def fail(*a,**k): raise OSError('discovery unavailable before admission')
    monkeypatch.setattr(bot_live_delivery,'read_delivery_result',fail)
    with executions._transaction() as conn:
        conn.execute("CREATE TRIGGER reject_release BEFORE UPDATE OF delivery_manifest_pending ON executions WHEN NEW.delivery_manifest_pending=0 BEGIN SELECT RAISE(ABORT,'transient intent release fault'); END")
    job,row=run(False,deliver='bot-chat')
    with executions._transaction() as conn: conn.execute('DROP TRIGGER reject_release')
    restart(home,count=2)
    result=snapshot(home,job,row,sent)
    print('NO_RECEIPT_RELEASE_FAULT',json.dumps(result,default=str))
    assert not sent and not result['receipts'] and not result['journal']
    assert result['row']['delivery_manifest_pending']==0, 'Storage repaired but no deferred release exists; permanently queued'


def test_nested_script_inner_intent_does_not_mark_outer(setup,monkeypatch):
    home,cfg,run,sent=setup
    driver=Path(__file__).with_name('_r5_nested_driver.py')
    (home/'scripts'/'outer.sh').write_text('#!/bin/sh\n'+sys.executable+' '+str(driver)+'\n')
    outer=jobs.create_job(prompt=None,schedule='every 1h',script='outer.sh',no_agent=True,deliver='local')
    row=executions.create_execution(outer['id'],source='external-outer')
    outer['execution_id']=row['id']
    monkeypatch.setenv('_HERMES_CRON_EXTERNAL_WORKER',row['id'])
    assert scheduler.run_one_job(outer)
    receipt=json.loads((home/'inner-proof.json').read_text())
    after=executions.get_execution(row['id'])
    print('NESTED_SCRIPT',json.dumps({'outer':after,'inner':receipt}))
    assert receipt['outer_env']==row['id']
    assert receipt['inner']['id'] != row['id']
    assert receipt['inner']['delivery_manifest_pending']==1
    assert after['delivery_manifest_pending']==0 and after['delivery_outcome']=='suppressed'  # outer script emits no output
    assert receipt['inner']['delivery_outcome']=='queued'
