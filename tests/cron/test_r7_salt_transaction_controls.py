"""Salt R7 transaction/RLock/Q2 controls, ported as regression cells."""
import json, sqlite3, subprocess, sys
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/"cron"/"executions.py").exists())
def _env(**extra):
    env=dict(os.environ); env["PYTHONPATH"]=str(REPO); env.update(extra); return env
from cron import executions as ex, incidents
from tests.cron.test_r5_salt_legacy_and_races import store
from tests.cron.test_r6_salt_migration_authority import legacy_running


def test_incident_upsert_cold_adoption_preserves_outer_transaction(store,monkeypatch,caplog):
    eid,manifest=legacy_running(store)
    incident,is_new=incidents.upsert_incident('child-job','r7 script failure',execution_id=eid)
    with sqlite3.connect(ex.EXECUTIONS_FILE) as conn:
        binding=conn.execute('SELECT incident_id,incident_generation FROM executions WHERE id=?',(eid,)).fetchone()
        assert binding[0]==incident and binding[1] is not None
        assert conn.execute('SELECT count(*) FROM cron_incidents WHERE id=?',(incident,)).fetchone()[0]==1
        assert conn.execute('SELECT count(*) FROM schema_meta').fetchone()[0]==0
    # Q2 FIXED: adoption is deferred quietly when the caller owns the transaction (no nested BEGIN attempt).
    assert 'cannot start a transaction within a transaction' not in caplog.text
    row=ex.get_execution(eid)
    assert ex._legacy_intent_adopted() and row['delivery_manifest_pending']==1
    print('INCIDENT_OUTER_TRANSACTION',json.dumps({'binding':binding,'new':is_new,'adopted_on_next_open':True}))


def test_pending_with_null_manifest_is_reachable_and_must_stay_queued(store):
    # Modern producer marks BEFORE it can store any manifest.
    row=ex.create_execution('null-pending',source='r7');eid=row['id']
    ex.mark_delivery_manifest_pending(eid)
    assert ex.get_execution(eid)['delivery_manifest'] is None
    p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('_r7_exit_child.py'))],env=_env());p.wait()
    with sqlite3.connect(ex.EXECUTIONS_FILE) as conn:
        conn.execute("UPDATE executions SET process_id='exited-r7',pid=?,process_started_at=NULL WHERE id=?",(p.pid,eid))
    assert ex.recover_interrupted_executions()==1
    row=ex.get_execution(eid)
    print('NULL_PENDING',json.dumps(row))
    assert row['delivery_manifest'] is None and row['delivery_manifest_pending']==1
    assert row['delivery_outcome']=='queued' and row['status']=='unknown'
