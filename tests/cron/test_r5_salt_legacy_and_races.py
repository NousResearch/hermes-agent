"""Salt R5 K2/S5 probes (legacy upgrade + reconciler races), ported as regression cells."""
import importlib.util, json, sqlite3, threading
from pathlib import Path
import pytest
from cron import executions as ex, jobs, delivery_queue as dq, bot_chat_delivery as bot
from utils import atomic_json_write
ROOT=Path(__file__).resolve().parent
SALT=Path('/home/deploy/work/nous368-full-audit/salt-r5-review')

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    monkeypatch.setenv('HERMES_MANAGED_DIR',str(tmp_path/'managed'))
    monkeypatch.setattr(ex,'EXECUTIONS_FILE',tmp_path/'cron'/'executions.db')
    monkeypatch.setattr(dq,'DELIVERY_DB',tmp_path/'cron'/'deliveries.db')
    return tmp_path

def prior():
    spec=importlib.util.spec_from_file_location('child_prior',ROOT/'_r5_prior_executions.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def seed(home, module=ex, placeholder=True):
    module.EXECUTIONS_FILE=home/'cron'/'executions.db'
    row=module.create_execution('child-job',source='evidence')
    eid=row['id']
    module.record_delivery_manifest(eid,{'external':True} if placeholder else {'bot':{}})
    module.finish_execution(eid,success=True,delivery_outcome='queued')
    # Actual durable jobs store, no scheduler/model/transport.
    jobs.save_jobs([{'id':'child-job','name':'child probe','prompt':'probe','enabled':False,'schedule':{'kind':'interval','minutes':60},'delivery_execution_id':eid,'last_status':'delivery_queued','last_delivery_queued':{'gateway':{'status':'queued'}}}])
    dq.enqueue(eid,{'id':'child-job'},'evidence')
    assert dq.claim_next()['execution_id']==eid
    assert dq._finish(eid,error=None)
    return eid

def receipt(home,eid):
    key=eid+'b07'
    bot.defer(key,{'id':'child-job'},'evidence','',home)
    return {'bot':{'bot-chat:(own)':{'home':str(home),'delivery_id':key,'status':'queued'}},'delivered':True}

def set_receipt(manifest,status):
    ref=next(iter(manifest['bot'].values()))
    row=bot.read_pending(ref['delivery_id']);row['status']=status
    atomic_json_write(bot._root()/(ref['delivery_id']+'.json'),row,fsync_dir=True,mode=0o600)

def snap(eid):
    return {'execution':ex.get_execution(eid),'job':jobs.get_job('child-job'),'journal':ex.journaled_manifests()}

def save(name,data):
    print(name,json.dumps(data))

@pytest.mark.parametrize('version',['prior','head'])
def test_legacy_failed_replay_must_not_project_placeholder(store,version):
    old=prior();eid=seed(store,old);manifest=receipt(store,eid)
    old.journal_manifest(eid,manifest)
    with sqlite3.connect(old.EXECUTIONS_FILE) as conn:
        assert 'delivery_manifest_pending' not in [r[1] for r in conn.execute('PRAGMA table_info(executions)')]
        conn.execute("CREATE TRIGGER child_reject_manifest BEFORE UPDATE OF delivery_manifest ON executions WHEN instr(NEW.delivery_manifest, '\"bot\"')>0 BEGIN SELECT RAISE(ABORT,'child persistent manifest fault'); END")
    target=old if version=='prior' else ex
    target.reconcile_delivery_projections()
    state=snap(eid)
    with sqlite3.connect(old.EXECUTIONS_FILE) as conn:
        conn.execute('DROP TRIGGER child_reject_manifest')
    set_receipt(manifest,'ambiguous')
    target.reconcile_delivery_projections()
    repaired=snap(eid)
    save('legacy-'+version,{'after_failed_replay':state,'after_repair':repaired})
    assert state['execution']['delivery_outcome']=='queued'
    assert state['execution']['delivery_projection_settled']==0
    assert state['job']['last_delivery_queued']

@pytest.mark.parametrize('kind',['no_manifest','external_only','healthy_replay'])
def test_legacy_default_zero_parity_controls(store,kind):
    old=prior()
    if kind=='no_manifest':
        eid=old.create_execution('child-job',source='evidence')['id']
        old.finish_execution(eid,success=True,delivery_outcome='not_configured')
        before=old.get_execution(eid)
        ex.reconcile_delivery_projections();after=ex.get_execution(eid)
        assert after['delivery_outcome']==before['delivery_outcome']
    else:
        eid=seed(store,old)
        if kind=='healthy_replay':old.journal_manifest(eid,receipt(store,eid))
        ex.reconcile_delivery_projections();after=ex.get_execution(eid)
        assert after['delivery_outcome']==('queued' if kind=='healthy_replay' else 'delivered')
    assert after['delivery_manifest_pending']==0
    save('parity-'+kind,after)


def test_two_reconcilers_repair_before_stale_projection_write(store,monkeypatch):
    eid=seed(store);manifest=receipt(store,eid)
    ex.mark_delivery_manifest_pending(eid)
    assert ex.journal_manifest(eid,manifest)
    reached=threading.Event();release=threading.Event();errors=[]
    original=ex._delivery_projection
    def barrier(record):
        result=original(record)
        if threading.current_thread().name=='child-stale':
            reached.set();assert release.wait(15)
        return result
    monkeypatch.setattr(ex,'_delivery_projection',barrier)
    def run():
        try:ex.reconcile_delivery_projections()
        except BaseException as err:errors.append(repr(err))
    t=threading.Thread(target=run,name='child-stale');t.start()
    assert reached.wait(15)
    repaired=ex.get_execution(eid)
    assert repaired['delivery_manifest_pending']==0
    set_receipt(manifest,'settled');ex.reconcile_delivery_projections()
    winner=snap(eid)
    release.set();t.join(15);assert not t.is_alive() and not errors
    after=snap(eid)
    save('race-before-ledger',{'winner':winner,'after':after})
    assert after['execution']['delivery_outcome']=='delivered'
    assert not after['job']['last_delivery_queued']
    assert not after['journal']


def test_two_reconcilers_repair_and_settle_between_ledger_and_jobs(store,monkeypatch):
    eid=seed(store);manifest=receipt(store,eid)
    ex.mark_delivery_manifest_pending(eid);assert ex.journal_manifest(eid,manifest)
    reached=threading.Event();release=threading.Event();errors=[]
    original=jobs.update_delivery_projection
    def barrier(*args,**kwargs):
        if threading.current_thread().name=='child-stale':
            reached.set();assert release.wait(15)
        return original(*args,**kwargs)
    monkeypatch.setattr(jobs,'update_delivery_projection',barrier)
    def run():
        try:ex.reconcile_delivery_projections()
        except BaseException as err:errors.append(repr(err))
    t=threading.Thread(target=run,name='child-stale');t.start()
    assert reached.wait(15)
    assert ex.get_execution(eid)['delivery_manifest_pending']==0
    set_receipt(manifest,'settled');ex.reconcile_delivery_projections()
    winner=snap(eid)
    release.set();t.join(15);assert not t.is_alive() and not errors
    after=snap(eid)
    monkeypatch.setattr(ex,'MAX_TERMINAL_EXECUTIONS',0)
    with ex._transaction() as conn:ex._prune_unlocked(conn)
    ex.reconcile_delivery_projections()
    pruned=snap(eid)
    save('race-after-ledger',{'winner':winner,'after':after,'after_prune':pruned})
    assert not after['job']['last_delivery_queued'], 'stale queued projection resurrected after winning terminal projection'
    assert not pruned['job']['last_delivery_queued']
