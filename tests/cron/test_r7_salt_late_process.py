"""Salt R7 K2c: real separate pre-flag writer publishing after adoption, ported as regression cell."""
import json, os, os, sqlite3, subprocess, sys, time
from pathlib import Path
import pytest
from cron import executions as ex, delivery_queue as dq, jobs
from tests.cron.test_r5_salt_legacy_and_races import store, prior, snap
HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'cron'/'executions.py').exists())
def _env(**extra):
    env=dict(os.environ); env['PYTHONPATH']=str(REPO); env.update(extra); return env

@pytest.mark.parametrize('replay_first',[False,True])
def test_old_process_publishes_after_adoption(store,replay_first):
    old=prior();old.EXECUTIONS_FILE=ex.EXECUTIONS_FILE
    eid=old.create_execution('child-job',source='r7-late-process')['id']
    old.record_delivery_manifest(eid,{'external':True})
    with old._transaction() as conn: conn.execute('UPDATE executions SET process_id=? WHERE id=?',(ex._PROCESS_ID,eid))
    jobs.save_jobs([{'id':'child-job','name':'r7','enabled':False,'schedule':{'kind':'interval','minutes':60},'delivery_execution_id':eid,'last_delivery_queued':{'gateway':{'status':'queued'}}}])
    dq.enqueue(eid,{'id':'child-job'},'evidence')
    barrier=store/'writer';barrier.mkdir()
    p=subprocess.Popen([sys.executable,str(HERE/'_r7_old_writer.py'),str(store),eid,str(barrier)],env=_env(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+20
        while not (barrier/'ready').exists():
            assert time.monotonic()<deadline and p.poll() is None
            time.sleep(.01)
        initial=ex.get_execution(eid)
        assert ex._legacy_intent_adopted() and initial['delivery_manifest_pending']==0
        assert dq.get_status(eid)['status']=='delivering'
        (barrier/'go').write_text('publish')
        stdout,stderr=p.communicate(timeout=20);assert p.returncode==0,(stdout,stderr)
        writer=json.loads(stdout.strip().splitlines()[-1])
        assert dq.get_status(eid)['status']=='delivered'
        assert ex.journaled_manifests()[eid]['bot']
        if replay_first: ex._recover_unrecorded_manifests()
        finished=ex.finish_execution(eid,success=True,delivery_outcome='queued')
        after=snap(eid)
        # Another fresh reader performs restart reconciliation; no cached authority helps.
        p2=subprocess.run([sys.executable,str(HERE/'_r7_restart_read.py'),str(store),eid],env=_env(),capture_output=True,text=True,timeout=20)
        assert p2.returncode==0,(p2.stdout,p2.stderr)
        restarted=json.loads(p2.stdout.strip().splitlines()[-1])
        print('LATE_PROCESS',json.dumps({'replay_first':replay_first,'writer':writer,'finished':finished,'after':after,'restarted':restarted}))
        assert restarted['execution']['delivery_outcome']=='queued', 'replayed queued child cannot correct terminal delivered'
    finally:
        if p.poll() is None:p.kill();p.wait()
