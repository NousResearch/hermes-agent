"""Salt R9 probes (K2e finish truth after lost adoption, S7 non-canonical placeholder replay convergence), ported as regression cells."""
import json, os, subprocess, sys
from pathlib import Path
import pytest
from cron import executions as ex, jobs, bot_chat_delivery as bot
from tests.cron.test_r5_salt_legacy_and_races import store, snap
from tests.cron.test_r8_salt_race_probes import adopted, child

def drain(monkeypatch, error=None):
    from cron import scheduler_delivery as sd
    from tools import bot_live_delivery as live
    calls=[]
    with monkeypatch.context() as m:
        m.setattr(live,'find_canonical_owner',lambda home:None)
        def send(*args,**kwargs):calls.append(kwargs['deferred']['id']);return error
        m.setattr(sd,'_deliver_to_bot_chat',send)
        bot._drain(bot._root())
    assert len(calls)==1

@pytest.mark.parametrize('schedule',['before_stat','during_stat'])
@pytest.mark.parametrize('error',[None,'independent downstream failure'])
def test_complete_row_fallback_truth(store,monkeypatch,schedule,error):
    eid,manifest,old=adopted(store);old.journal_manifest(eid,manifest)
    if schedule=='before_stat':child('replay',store,eid)
    original=ex._journal_entry_state;seen=[]
    def barrier(key):
        state=original(key)
        if not seen:
            assert state is True
            seen.append(child('replay',store,eid))
        return state
    with monkeypatch.context() as m:
        m.setattr(ex,'_journal_entry_state',barrier)
        finished=ex.finish_execution(eid,success=True,delivery_outcome='queued')
    before=snap(eid)
    assert before['execution']['delivery_manifest_pending']==0
    assert 'bot' in json.loads(before['execution']['delivery_manifest'])
    assert bot.read_pending(next(iter(manifest['bot'].values()))['delivery_id'])['status']=='queued'
    drain(monkeypatch,error)
    restarted=child('read',store,eid)
    expected='unknown' if error else 'delivered'
    print('FALLBACK_TRUTH',json.dumps(dict(schedule=schedule,error=error,finished=finished,before=before,restart=restarted,expected=expected)))
    checks={'queued_before_child':before['execution']['delivery_outcome']=='queued',
            'truth_after_restart':restarted['execution']['delivery_outcome']==expected,
            'projection_settled':restarted['execution']['delivery_projection_settled']==1}
    assert all(checks.values()), checks

@pytest.mark.parametrize('replacement',[{'external':True},{'external':True,'delivered':False}])
def test_replaced_placeholder_adoption_and_replay(store,monkeypatch,replacement):
    eid,manifest,old=adopted(store);old.journal_manifest(eid,manifest)
    original=ex._journal_entry_state;seen=[]
    def barrier(key):
        state=original(key)
        if not seen:
            ex.record_delivery_manifest(eid,replacement)
            seen.append(ex.get_execution(eid))
        return state
    with monkeypatch.context() as m:
        m.setattr(ex,'_journal_entry_state',barrier)
        m.setattr(ex,'_recover_unrecorded_manifests',lambda:{})
        finished=ex.finish_execution(eid,success=True,delivery_outcome='queued')
    assert finished['delivery_manifest_pending']==1
    drain(monkeypatch)
    restarts=[child('read',store,eid) for _ in range(2)]
    print('REPLACEMENT',json.dumps(dict(replacement=replacement,seen=seen,finished=finished,restarts=restarts)))
    assert restarts[-1]['execution']['delivery_outcome']=='delivered', 'shape accepted for adoption is rejected by manifest replacement CAS'
    assert restarts[-1]['execution']['delivery_manifest_pending']==0
    assert restarts[-1]['execution']['delivery_projection_settled']==1

@pytest.mark.parametrize('existing',[{'external':True},{'external':True,'delivered':False}])
def test_projection_only_replay_converges(store,monkeypatch,existing):
    eid,manifest,old=adopted(store)
    ex.record_delivery_manifest(eid,existing)
    old.journal_manifest(eid,manifest)
    row=ex.get_execution(eid)
    assert row['delivery_manifest_pending']==0
    assert ex._delivery_projection(row) is None
    assert ex.get_execution(eid)['delivery_manifest_pending']==0
    # Fixture terminal run, not a projection stub: projection is a read-only function.
    with ex._transaction() as conn:
        conn.execute("UPDATE executions SET status='completed', delivery_outcome='queued' WHERE id=?",(eid,))
    drain(monkeypatch)
    first=child('read',store,eid);second=child('read',store,eid)
    print('PROJECTION_REPLAY',json.dumps(dict(existing=existing,first=first,second=second)))
    assert second['execution']['delivery_manifest_pending']==0
    assert second['execution']['delivery_outcome']=='delivered'
    assert second['execution']['delivery_projection_settled']==1
