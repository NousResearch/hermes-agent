"""Salt R6 controls, ported: healthy adoption, freshness, settlement durability; Q1 secondary-exception contract now FIXED (original wins)."""
import json
import sqlite3
import sys
from pathlib import Path
import pytest
from tests.cron.test_r6_salt_migration_authority import legacy_running
from tests.cron.test_r5_salt_legacy_and_races import store, seed
from tests.cron.test_mixed_delivery_retention import setup
from cron import executions as ex, jobs, scheduler_delivery as sd


def test_healthy_adoption_precedes_late_finish(store):
    eid,manifest=legacy_running(store)
    row=ex.get_execution(eid)
    assert row['delivery_manifest_pending']==1
    row=ex.finish_execution(eid,success=True,delivery_outcome='queued')
    assert row['delivery_outcome']=='queued'
    assert ex.get_execution(eid)['delivery_outcome']=='queued'


def test_new_execution_after_unknown_accepts_queued(store):
    eid=seed(store)
    with ex._transaction() as conn:
        conn.execute("UPDATE executions SET delivery_outcome='unknown' WHERE id=?",(eid,))
    newer=ex.create_execution('child-job',source='r6-explicit-new-run')
    jobs.bind_delivery_execution('child-job',newer['id'])
    values={'last_delivery_queued':{'bot-chat:test':{'status':'queued'}}}
    assert jobs.update_delivery_projection('child-job',newer['id'],values)
    assert jobs.get_job('child-job')['last_delivery_queued']==values['last_delivery_queued']


def test_original_exception_survives_all_ordinary_bookkeeping_faults(setup,monkeypatch):
    home,cfg,run,sent=setup
    job=jobs.create_job(prompt='probe',schedule='every 1h',deliver='telegram:test,bot-chat')
    row=ex.create_execution(job['id'],source='r6');job['execution_id']=row['id']
    jobs.bind_delivery_execution(job['id'],row['id'])
    original=sd._deliver_targets
    error=RuntimeError('original post-admission fault')
    def fail_after(*a,**k):
        original(*a,**k)
        raise error
    monkeypatch.setattr(sd,'_deliver_targets',fail_after)
    def fail_jobs(*a,**k): raise OSError('jobs write refused')
    monkeypatch.setattr(jobs,'update_delivery_projection',fail_jobs)
    with ex._transaction() as conn:
        conn.execute("CREATE TRIGGER r6_settle_fault BEFORE UPDATE OF delivery_manifest ON executions BEGIN SELECT RAISE(ABORT,'manifest write refused'); END")
    root=ex._journal_root();root.mkdir();root.chmod(0)
    try:
        with pytest.raises(RuntimeError) as caught: sd._deliver_result(job,'r6 payload')
        assert caught.value is error
        assert len(sent)==1 and job['_bot_chat_delivery_receipts']
        assert ex.get_execution(row['id'])['delivery_manifest_pending']==1
    finally:
        root.chmod(0o700)


def test_secondary_baseexception_does_not_mask_original(setup,monkeypatch):
    import asyncio
    home,cfg,run,sent=setup
    job=jobs.create_job(prompt='probe',schedule='every 1h',deliver='bot-chat')
    row=ex.create_execution(job['id'],source='r6');job['execution_id']=row['id']
    original=OSError('original transport discovery failure')
    secondary=asyncio.CancelledError('second interruption during manifest persistence')
    def fail_targets(*a,**k): raise original
    def fail_record(*a,**k): raise secondary
    monkeypatch.setattr(sd,'_deliver_targets',fail_targets)
    monkeypatch.setattr(ex,'record_delivery_manifest',fail_record)
    with pytest.raises(OSError) as caught: sd._deliver_result(job,'r6 payload')
    assert caught.value is original
    assert ex.get_execution(row['id'])['delivery_manifest_pending']==1  # uncertainty preserved


def test_column_presence_is_not_the_authority_fence_row_is(store):
    """DDL commits on its own; the schema_meta fence row is what readers trust."""
    eid,manifest=legacy_running(store)
    assert ex.get_execution(eid)['delivery_manifest_pending']==1
    with sqlite3.connect(ex.EXECUTIONS_FILE) as other:
        assert other.execute("SELECT 1 FROM schema_meta WHERE key=?",(ex._ADOPTION_KEY,)).fetchone()
    assert ex._legacy_intent_adopted()
