"""Salt R3 B2: actual SQLite write rejection persists across fresh reconcilers."""
import json, sqlite3
from cron import executions, scheduler, jobs, bot_chat_delivery
from tests.cron.test_mixed_delivery_retention import setup
from tests.cron.test_r4_salt_recovery_probes import restart


def test_actual_sqlite_manifest_update_failure_keeps_pending_sibling(setup):
    home,cfg,run,sent=setup
    # Initialize real database; reject ONLY full-manifest updates, not placeholder/projection.
    with executions._transaction() as conn:
        conn.execute('''CREATE TRIGGER salt_reject_manifest BEFORE UPDATE OF delivery_manifest ON executions
          WHEN NEW.delivery_manifest LIKE '%"bot"%'
          BEGIN SELECT RAISE(ABORT, 'salt persistent manifest storage fault'); END''')
    job,before=run(True);scheduler.drain_delivery_queue({},None)
    restart(home,count=2) # no Python fault hook: SQLite itself rejects replay after restart.
    mid=executions.get_execution(before['id'])
    receipt=bot_chat_delivery._records(bot_chat_delivery._root())[0][1]
    print('SQLITE_FAULT',json.dumps({'execution':mid,'receipt_status':receipt['status'],'journaled':before['id'] in executions.journaled_manifests(),'queued':jobs.get_job(job['id']).get('last_delivery_queued'),'native_sends':len(sent)}))
    with sqlite3.connect(home/'cron/executions.db') as conn:conn.execute('DROP TRIGGER salt_reject_manifest')
    restart(home)
    recovered=executions.get_execution(before['id'])
    print('SQLITE_REPAIRED',json.dumps({'execution':recovered,'queued':jobs.get_job(job['id']).get('last_delivery_queued')}))
    assert mid['delivery_outcome']=='queued'
    assert recovered['delivery_outcome']=='queued'
