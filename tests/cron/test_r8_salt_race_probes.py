"""Salt R8 probes (after-stat writer barriers, K2d replay-between-stat-and-write, stat failure matrix, reopen predicate, CAS), ported as regression cells."""
import errno, json, os, sqlite3, subprocess, sys, time
from pathlib import Path
import pytest
from cron import executions as ex, delivery_queue as dq, jobs
from tests.cron.test_r5_salt_legacy_and_races import store, prior, receipt, snap, set_receipt
from tests.cron.test_r6_salt_migration_authority import legacy_running
HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'cron'/'executions.py').exists())
def _env(**extra):
    env=dict(os.environ); env['PYTHONPATH']=str(REPO); env.update(extra); return env

def child(mode,store,eid):
    p=subprocess.run([sys.executable,str(HERE/'_r8_child.py'),mode,str(store),eid],env=_env(),capture_output=True,text=True,timeout=25)
    assert p.returncode==0,(p.stdout,p.stderr)
    return json.loads(p.stdout.strip().splitlines()[-1])

def adopted(store):
    eid,manifest=legacy_running(store)
    (ex._journal_root()/f'{eid}.json').unlink()
    assert ex.get_execution(eid)['delivery_manifest_pending']==0
    assert ex._legacy_intent_adopted()
    old=prior(); old.EXECUTIONS_FILE=ex.EXECUTIONS_FILE
    return eid,manifest,old

def wait(path,p):
    deadline=time.monotonic()+20
    while not path.exists():
        assert p.poll() is None and time.monotonic()<deadline
        time.sleep(.005)

@pytest.mark.parametrize('publish_at',[1,2])
def test_writer_after_absent_stat_before_external_completes(store,monkeypatch,publish_at):
    old=prior();old.EXECUTIONS_FILE=ex.EXECUTIONS_FILE
    eid=old.create_execution('child-job',source='r8-barrier')['id']
    old.record_delivery_manifest(eid,{'external':True})
    with old._transaction() as conn:conn.execute('UPDATE executions SET process_id=? WHERE id=?',(ex._PROCESS_ID,eid))
    jobs.save_jobs([{'id':'child-job','enabled':False,'schedule':{'kind':'interval','minutes':60},'delivery_execution_id':eid,'last_delivery_queued':{'gateway':{'status':'queued'}}}])
    dq.enqueue(eid,{'id':'child-job'},'evidence')
    barrier=store/'barrier';barrier.mkdir()
    p=subprocess.Popen([sys.executable,str(HERE/'_r8_child.py'),'writer',str(store),eid,str(barrier)],env=_env(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        wait(barrier/'ready',p)
        assert ex.get_execution(eid)['delivery_manifest_pending']==0
        original=ex._journal_entry_state;calls=[]
        def stat_then_publish(key):
            state=original(key); calls.append(state)
            if len(calls)==publish_at:
                assert state is False
                (barrier/'publish').write_text('go');wait(barrier/'published',p)
                assert dq.get_status(eid)['status']=='delivering'
            return state
        with monkeypatch.context() as m:
            m.setattr(ex,'_journal_entry_state',stat_then_publish)
            finished=ex.finish_execution(eid,success=True,delivery_outcome='queued')
        assert finished['delivery_outcome']=='queued'
        (barrier/'complete').write_text('go')
        stdout,stderr=p.communicate(timeout=20);assert p.returncode==0,(stdout,stderr)
        restarted=child('read',store,eid)
        print('AFTER_STAT',json.dumps({'publish_at':publish_at,'stats':calls,'finished':finished,'restart':restarted}))
        assert restarted['execution']['delivery_outcome']=='queued'
        assert restarted['execution']['delivery_projection_settled']==0
        assert restarted['job']['last_delivery_queued']
    finally:
        if p.poll() is None:p.kill();p.wait()


@pytest.mark.parametrize('schedule',['before_stat','during_stat'])
def test_replay_between_stat_and_finish_intent_does_not_strand_complete_manifest(store,monkeypatch,schedule):
    eid,manifest,old=adopted(store)
    old.journal_manifest(eid,manifest)
    if schedule=='before_stat': child('replay',store,eid)
    original=ex._journal_entry_state; observed=[]; transaction_states=[]
    original_fetch=ex._fetch
    def traced_fetch(conn,key):
        transaction_states.append(conn.in_transaction)
        return original_fetch(conn,key)
    monkeypatch.setattr(ex,'_fetch',traced_fetch)
    def after_stat(key):
        result=original(key)
        if not observed:
            assert result is True
            replay=child('replay',store,eid)
            assert replay['execution']['delivery_manifest_pending']==0
            assert json.loads(replay['execution']['delivery_manifest'])['bot']
            observed.append(replay)
        return result
    with monkeypatch.context() as m:
        m.setattr(ex,'_journal_entry_state',after_stat)
        finished=ex.finish_execution(eid,success=True,delivery_outcome='queued')
    before=snap(eid)
    # Actual queue claim/settle implementation; only owner discovery and final turn are doubles.
    from cron import bot_chat_delivery as bot, scheduler_delivery as sd
    from tools import bot_live_delivery as live
    sends=[]
    with monkeypatch.context() as m:
        m.setattr(live,'find_canonical_owner',lambda home:None)
        def delivered(*args,**kwargs):sends.append(kwargs['deferred']['id']);return None
        m.setattr(sd,'_deliver_to_bot_chat',delivered)
        bot._drain(bot._root())
    assert len(sends)==1
    assert bot.read_pending(next(iter(manifest['bot'].values()))['delivery_id'])['status']=='settled'
    restarted=child('read',store,eid)
    print('REPLAY_DURING_STAT',json.dumps({'schedule':schedule,'transaction_states':transaction_states,'replay':observed,'finished':finished,'before':before,'restart':restarted}))
    assert restarted['execution']['delivery_outcome']=='delivered', 'late adoption reasserted pending on a complete manifest and stranded settlement'
    assert restarted['execution']['delivery_manifest_pending']==0
    assert restarted['execution']['delivery_projection_settled']==1

@pytest.mark.parametrize('shape',['directory_no_search','entry_no_read','missing_root','eio','bot_no_probe'])
def test_stat_failure_matrix(store,monkeypatch,shape):
    eid,manifest,old=adopted(store)
    root=ex._journal_root(); path=root/f'{eid}.json'
    expected={'directory_no_search':None,'entry_no_read':True,'missing_root':False,'eio':None}
    old.journal_manifest(eid,manifest)
    try:
        if shape=='directory_no_search':root.chmod(0o600)
        elif shape=='entry_no_read':path.chmod(0)
        elif shape=='missing_root':path.unlink();root.rmdir()
        elif shape=='eio':
            original=Path.stat
            def fail(p,*a,**kw):
                if p==path:raise OSError(errno.EIO,'injected stat I/O error')
                return original(p,*a,**kw)
            monkeypatch.setattr(Path,'stat',fail)
        else:
            ex.record_delivery_manifest(eid,manifest)
            def forbidden(*a):raise AssertionError('bot-complete row probed journal')
            monkeypatch.setattr(ex,'_journal_entry_state',forbidden)
            assert ex._late_legacy_intent(ex.get_execution(eid)) is False
            return
        state=ex._journal_entry_state(eid)
        assert state is expected[shape]
        with monkeypatch.context() as m:
            m.setattr(ex,'_recover_unrecorded_manifests',lambda:{})
            finished=ex.finish_execution(eid,success=True,delivery_outcome='queued')
        print('STAT_MATRIX',shape,json.dumps({'state':state,'finished':finished}))
        assert finished['delivery_outcome']==('delivered' if state is False else 'queued')
        assert finished['delivery_manifest_pending']==int(state is True)
    finally:
        if shape=='directory_no_search':root.chmod(0o700)
        if shape=='entry_no_read':path.chmod(0o600)

@pytest.mark.parametrize('outcome',['suppressed','not_configured','unknown','delivered','failed'])
@pytest.mark.parametrize('complete',[False,True])
def test_reopen_predicate(store,outcome,complete):
    eid,manifest,old=adopted(store)
    if complete:ex.record_delivery_manifest(eid,{'bot':{},'delivered':True})
    with ex._transaction() as conn:
        conn.execute("UPDATE executions SET status='completed',delivery_outcome=?,delivery_projection_settled=1 WHERE id=?",(outcome,eid))
    old.journal_manifest(eid,manifest)
    ex._recover_unrecorded_manifests()
    row=ex.get_execution(eid)
    should_reopen=not complete and outcome in ('delivered','failed')
    print('REOPEN_PREDICATE',outcome,complete,json.dumps(row))
    assert row['delivery_outcome']==('queued' if should_reopen else outcome)
    assert row['delivery_projection_settled']==(0 if should_reopen else 1)


def test_finish_adopted_flag_blocks_stale_reconciler_before_replay(store,monkeypatch):
    eid,manifest,old=adopted(store)
    stale=ex.get_execution(eid)
    projection=ex._delivery_projection(stale);assert projection[0]=='delivered'
    old.journal_manifest(eid,manifest)
    with monkeypatch.context() as m:
        m.setattr(ex,'_recover_unrecorded_manifests',lambda:{})
        finished=ex.finish_execution(eid,success=True,delivery_outcome='queued')
        assert finished['delivery_manifest_pending']==1
        m.setattr(ex,'_delivery_projection',lambda record: projection)
        ex.reconcile_delivery_projections()
    row=ex.get_execution(eid)
    print('CAS_AFTER_FINISH',json.dumps(row))
    assert row['delivery_outcome']=='queued' and row['delivery_projection_settled']==0
    assert row['delivery_manifest_pending']==1
    assert jobs.get_job('child-job')['last_delivery_queued']
