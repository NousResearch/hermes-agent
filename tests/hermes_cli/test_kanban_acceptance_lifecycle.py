"""Machine acceptance across the review lifecycle: reviewer handoff, operator re-completion, reopen and reaped workers."""
import subprocess
import pytest
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_acceptance as kba, kanban_acceptance as ka, kanban_db_dispatch as kbd

@pytest.fixture
def c(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path/'home'))
    monkeypatch.setenv('HERMES_GATEWAY_LOCK_DIR',str(tmp_path/'locks'))
    monkeypatch.setattr(kb,'_fire_task_hook',lambda *a,**kw:None)
    monkeypatch.setattr(ka,'_estop_allows',lambda:True)
    conn=kbc.connect(tmp_path/'kanban.db')
    yield conn
    conn.close()

def claim_exited(c,tid):
    t=kb.claim_task(c,tid)
    p=subprocess.Popen(['/usr/bin/sleep','30'])
    try:
        kbd._set_worker_pid(c,tid,p.pid,run_id=t.current_run_id,claim_lock=t.claim_lock)
        row=c.execute('SELECT worker_pid,worker_started_at FROM task_runs WHERE id=?',(t.current_run_id,)).fetchone()
        assert row['worker_pid']==p.pid and row['worker_started_at']
        p.terminate(); p.wait(timeout=10)
        assert not kb._pid_alive(p.pid)
    finally:
        if p.poll() is None: p.kill(); p.wait()
    return t

def test_closed_unknown_claimed_run_is_not_no_origin_worker(c,tmp_path):
    tid=kb.create_task(c,title='unknown',assignee='builder',acceptance={'kind':'file','path':str(tmp_path)})
    run=kb.claim_task(c,tid)  # deliberately no registration: identity not known
    assert kb.complete_task(c,tid,summary='implemented',expected_run_id=run.current_run_id)
    assert ka.run_acceptance_pass(c)[0]['reason']=='identity_unknown'
    assert kb.reopen_review_task(c,tid)
    assert kb.complete_task(c,tid,summary='operator resubmit',force=True)
    result=ka.run_acceptance_pass(c)
    assert result[0]['verdict']=='pending', result
    assert kb.get_task(c,tid).status=='review'

def test_reopen_machine_wait_restores_current_submission_implementer(c):
    tid=kb.create_task(c,title='rerouted',assignee='builder',acceptance={'kind':'tbd'})
    old=claim_exited(c,tid)
    assert kb.request_review(c,tid,summary='old review',reviewer='reviewer',expected_run_id=old.current_run_id)
    assert kb.reopen_review_task(c,tid)
    assert kb.assign_task(c,tid,'new-builder')
    new=claim_exited(c,tid)
    assert kb.complete_task(c,tid,summary='new impl',expected_run_id=new.current_run_id)
    assert kba.open_submission(c,tid)['implementer']=='new-builder'
    assert kb.reopen_review_task(c,tid)
    assert kb.get_task(c,tid).assignee=='new-builder'

def test_never_claimed_operator_completion_still_passes(c,tmp_path):
    tid=kb.create_task(c,title='operator only',assignee='builder',acceptance={'kind':'file','path':str(tmp_path)})
    assert kb.complete_task(c,tid,summary='operator work')
    assert ka.run_acceptance_pass(c)[0]['verdict']=='pass'




@pytest.mark.parametrize('mode', ['operator', 'claimed-reviewer'])
def test_unknown_implementer_manual_approval_cannot_prove_exit(c,tmp_path,mode):
    tid=kb.create_task(c,title='unknown implementation then approval',assignee='builder',acceptance={'kind':'file','path':str(tmp_path)})
    origin=kb.claim_task(c,tid)
    p=subprocess.Popen(['/usr/bin/sleep','30'])  # deliberately not registered
    try:
        assert kb.request_review(c,tid,summary='implemented',reviewer='reviewer',expected_run_id=origin.current_run_id)
        if mode=='claimed-reviewer':
            reviewer=kb.claim_review_task(c,tid)
            assert reviewer is not None
            rp=subprocess.Popen(['/usr/bin/sleep','30'])
            try:
                kbd._set_worker_pid(c,tid,rp.pid,run_id=reviewer.current_run_id,claim_lock=reviewer.claim_lock)
                rp.terminate(); rp.wait(timeout=10)
                assert not kb._pid_alive(rp.pid)
                assert kb.complete_task(c,tid,summary='reviewer approves',expected_run_id=reviewer.current_run_id)
            finally:
                if rp.poll() is None: rp.kill(); rp.wait(timeout=10)
        else:
            assert kb.complete_task(c,tid,summary='operator approves')
        result=ka.run_acceptance_pass(c)
        assert p.poll() is None
        assert result[0]['verdict']=='pending', result
        assert result[0]['reason']=='identity_unknown', result
        assert kb.get_task(c,tid).status=='review'
    finally:
        p.terminate(); p.wait(timeout=10)


@pytest.mark.parametrize('registration', ['spawned','worker_registered'])
@pytest.mark.parametrize('mode', ['reopen-force','manual-approval'])
def test_exited_reaped_worker_is_not_stuck(c,tmp_path,registration,mode):
    tid=kb.create_task(c,title='exited origin',assignee='builder',acceptance={'kind':'file','path':str(tmp_path)})
    run=kb.claim_task(c,tid)
    p=subprocess.Popen(['/usr/bin/sleep','30'])
    try:
        if registration=='spawned':
            kbd._set_worker_pid(c,tid,p.pid,run_id=run.current_run_id,claim_lock=run.claim_lock)
        else:
            assert kbd.adopt_worker_pid(c,tid,run.current_run_id,p.pid)
        p.terminate(); p.wait(timeout=10)
        assert not kb._pid_alive(p.pid)
        if mode=='manual-approval':
            assert kb.request_review(c,tid,summary='implemented',reviewer='reviewer',expected_run_id=run.current_run_id)
        else:
            assert kb.complete_task(c,tid,summary='implemented',expected_run_id=run.current_run_id)
        with kb.write_txn(c):
            c.execute('UPDATE task_runs SET ended_at=ended_at-600 WHERE id=?',(run.current_run_id,))
        kbd.reap_terminal_workers(c)
        assert kb.get_run(c,run.current_run_id).worker_pid is None
        if mode=='reopen-force':
            assert kb.reopen_review_task(c,tid)
        assert kb.complete_task(c,tid,summary='operator completion',force=True)
        sub=kba.open_submission(c,tid)
        assert sub['origin_kind']=='worker' and sub['worker_pid']==p.pid and sub['worker_fingerprint']
        result=ka.run_acceptance_pass(c)
        assert result[0]['verdict']=='pass',result
        assert kb.get_task(c,tid).status=='done'
    finally:
        if p.poll() is None: p.kill(); p.wait(timeout=10)


def test_ordinary_manual_review_reopen_unchanged(c):
    tid=kb.create_task(c,title='ordinary review',assignee='builder')
    run=claim_exited(c,tid)
    assert kb.request_review(c,tid,summary='implemented',reviewer='reviewer',expected_run_id=run.current_run_id)
    assert kb.get_task(c,tid).assignee=='reviewer'
    assert not kba.is_machine_wait(c,tid)
    assert kb.reopen_review_task(c,tid)
    task=kb.get_task(c,tid)
    assert (task.status,task.assignee)==('ready','builder')
