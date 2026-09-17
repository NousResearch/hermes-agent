"""Salt R7 probes (late legacy journal K2c, concurrent initializers, interrupted CASE matrix), ported as regression cells."""
import json, os, os, sqlite3, subprocess, sys, time
from pathlib import Path
import pytest
from cron import executions as ex, jobs
from tests.cron.test_r5_salt_legacy_and_races import store, prior, receipt, snap
from tests.cron.test_r6_salt_migration_authority import legacy_running
from tests.cron.test_manifest_intent_upgrade import _reject_bot_manifest
HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'cron'/'executions.py').exists())
def _env(**extra):
    env=dict(os.environ); env['PYTHONPATH']=str(REPO); env.update(extra); return env

@pytest.mark.parametrize('window',['after_commit','after_inventory'])
@pytest.mark.parametrize('replay_first',[False,True])
def test_late_old_journal_before_finish(store,monkeypatch,window,replay_first):
    eid,manifest=legacy_running(store)
    journal=ex._journal_root()/(eid+'.json'); journal.unlink()
    old=prior(); old.EXECUTIONS_FILE=ex.EXECUTIONS_FILE
    if window=='after_inventory':
        original=ex._journal_inventory_strict
        def after_snapshot():
            result=original()
            old.journal_manifest(eid,manifest)
            return result
        monkeypatch.setattr(ex,'_journal_inventory_strict',after_snapshot)
    initial=ex.get_execution(eid)
    assert ex._legacy_intent_adopted() and initial['delivery_manifest_pending']==0
    if window=='after_commit': old.journal_manifest(eid,manifest)
    # No pending reset or production-projection stub. Actual old journal writer lacks preflag.
    if replay_first:
        with ex._transaction() as conn: _reject_bot_manifest(conn)
        ex._recover_unrecorded_manifests()
        assert ex.get_execution(eid)['delivery_manifest_pending']==1
        with ex._transaction() as conn: conn.execute('DROP TRIGGER k2_reject')
    finished=ex.finish_execution(eid,success=True,delivery_outcome='queued')
    after=snap(eid)
    print('LATE_JOURNAL',json.dumps({'window':window,'replay_first':replay_first,'initial':initial,'finished':finished,'after':after}))
    assert after['execution']['delivery_outcome']=='queued', 'late old journal escaped adoption; replay cannot undo false terminal outcome'
    assert jobs.get_job('child-job')['last_delivery_queued']

@pytest.mark.parametrize('iteration',range(3))
def test_two_concurrent_old_schema_initializers(store,iteration):
    eid,manifest=legacy_running(store)
    barrier=store/'barrier'; barrier.mkdir()
    processes=[]
    try:
        for name in ('a','b'):
            env=_env(HERMES_HOME=str(store))
            p=subprocess.Popen([sys.executable,str(HERE/'_r7_initializer_worker.py'),str(store),str(barrier),name],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            processes.append(p)
        deadline=time.monotonic()+20
        while not all((barrier/(name+'.ready')).exists() for name in ('a','b')):
            assert time.monotonic()<deadline, 'concurrent initialization never reached inventory boundary'
            assert all(p.poll() is None for p in processes), 'initializer exited early'
            time.sleep(.01)
        (barrier/'go').write_text('go')
        results=[]
        for p in processes:
            stdout,stderr=p.communicate(timeout=25)
            assert p.returncode==0,(stdout,stderr)
            result=json.loads(stdout.strip().splitlines()[-1]);results.append(result)
            assert result['adopted'] and result['pending']==[1]
            assert result['entry'][0]['in_transaction'] is False
            # Journal mode is runtime policy (hermes_state_wal gates WAL by SQLite build), not the
            # contract under test; the atomicity evidence below is what one adoption must satisfy.
            assert result['entry'][0]['journal_mode'] in ('wal','delete')
        assert sum('COMMIT' in r['trace'] for r in results)==1
        assert sum('ROLLBACK' in r['trace'] for r in results)==1
        with sqlite3.connect(ex.EXECUTIONS_FILE) as conn:
            assert conn.execute('SELECT count(*) FROM schema_meta WHERE key=?',(ex._ADOPTION_KEY,)).fetchone()[0]==1
            assert conn.execute('SELECT delivery_manifest_pending FROM executions WHERE id=?',(eid,)).fetchone()[0]==1
        print('TWO_INITIALIZERS',iteration,json.dumps(results))
    finally:
        for p in processes:
            if p.poll() is None: p.kill();p.wait()

@pytest.mark.parametrize('adoption_complete',[False,True])
@pytest.mark.parametrize('manifest',[None,{'external':True},{'bot':{}}],ids=['null','placeholder','bot'])
@pytest.mark.parametrize('flag',[0,1])
def test_interrupted_case_matrix(store,monkeypatch,adoption_complete,manifest,flag):
    old=prior();old.EXECUTIONS_FILE=ex.EXECUTIONS_FILE
    eid=old.create_execution('matrix',source='r7')['id']
    if manifest is not None: old.record_delivery_manifest(eid,manifest)
    root=ex._journal_root();root.mkdir(exist_ok=True)
    if not adoption_complete: root.chmod(0)
    try:
        ex.get_execution(eid)
        assert ex._legacy_intent_adopted() is adoption_complete
        # Real exited process identity, not a monkeypatched liveness result.
        # pid selected from a Popen whose exit was waited, not guessed.
        p=subprocess.Popen([sys.executable,str(HERE/'_r7_exit_child.py')],env=_env());p.wait()
        with sqlite3.connect(ex.EXECUTIONS_FILE) as conn:
            conn.execute("UPDATE executions SET process_id='gone-r7',pid=?,process_started_at=NULL,delivery_manifest_pending=?,delivery_outcome=NULL WHERE id=?",(p.pid,flag,eid))
        assert ex.recover_interrupted_executions()==1
        row=ex.get_execution(eid)
        expected='queued' if flag or (not adoption_complete and manifest is not None and 'bot' not in manifest) else None
        print('RECOVERY_CASE',json.dumps({'adopted':adoption_complete,'manifest':manifest,'flag':flag,'outcome':row['delivery_outcome']}))
        assert row['status']=='unknown' and row['delivery_outcome']==expected
    finally: root.chmod(0o700)


def test_external_only_liveness_and_prune_connection(store,monkeypatch):
    old=prior();old.EXECUTIONS_FILE=ex.EXECUTIONS_FILE
    eid=old.create_execution('external-only',source='r7')['id']
    old.record_delivery_manifest(eid,{'external':True})
    with old._transaction() as conn: conn.execute('UPDATE executions SET process_id=? WHERE id=?',(ex._PROCESS_ID,eid))
    root=ex._journal_root();root.mkdir(exist_ok=True);root.chmod(0)
    try:
        row=ex.finish_execution(eid,success=False,delivery_outcome='failed')
        assert row['delivery_outcome']=='queued'
        assert ex._delivery_projection(row) is None
        with ex._transaction() as conn:
            conn.execute("UPDATE executions SET delivery_outcome='delivered',delivery_projection_settled=1 WHERE id=?",(eid,))
            statements=[];conn.set_trace_callback(statements.append)
            def no_new_connection(): raise AssertionError('nested connection opened by prune')
            with monkeypatch.context() as m:
                m.setattr(ex,'MAX_TERMINAL_EXECUTIONS',0)
                m.setattr(ex,'_connect',no_new_connection)
                ex._prune_unlocked(conn)
            assert ex._fetch(conn,eid) is not None
            assert not any(s.startswith('BEGIN') for s in statements)
        print('PRUNE_CONNECTION',json.dumps(statements))
        assert not ex._legacy_intent_adopted()
    finally: root.chmod(0o700)
    assert ex._legacy_intent_adopted()
    with ex._transaction() as conn:
        with monkeypatch.context() as m:
            m.setattr(ex,'MAX_TERMINAL_EXECUTIONS',0);ex._prune_unlocked(conn)
    assert ex.get_execution(eid) is None


def test_multirow_adoption_rolls_back_all_and_retries(store):
    eid,manifest=legacy_running(store)
    old=prior();old.EXECUTIONS_FILE=ex.EXECUTIONS_FILE
    second=old.create_execution('second',source='r7')['id'];old.record_delivery_manifest(second,{'external':True});old.journal_manifest(second,receipt(store,second))
    last=max(eid,second)
    with sqlite3.connect(ex.EXECUTIONS_FILE) as conn:
        conn.execute(f"CREATE TRIGGER r7_partial BEFORE UPDATE ON executions WHEN NEW.id='{last}' AND NEW.delivery_manifest_pending=1 BEGIN SELECT RAISE(ABORT,'second row fault'); END")
    ex.get_execution(eid)
    with sqlite3.connect(ex.EXECUTIONS_FILE) as conn:
        assert conn.execute('SELECT count(*) FROM schema_meta').fetchone()[0]==0
        assert conn.execute('SELECT sum(delivery_manifest_pending) FROM executions').fetchone()[0]==0
        conn.execute('DROP TRIGGER r7_partial')
    ex.get_execution(eid)
    with sqlite3.connect(ex.EXECUTIONS_FILE) as conn:
        assert conn.execute('SELECT count(*) FROM schema_meta').fetchone()[0]==1
        assert conn.execute('SELECT sum(delivery_manifest_pending) FROM executions').fetchone()[0]==2
    print('ATOMIC_ADOPTION','rollback all rows and fence, retry both rows')
