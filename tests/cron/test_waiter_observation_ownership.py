"""A gateway observer cannot retire a restart-safe worker's durable claim."""
import sqlite3
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize('error', [sqlite3.OperationalError('database is locked'), RuntimeError('observer failed')])
def test_post_handoff_observation_error_preserves_live_claim(tmp_path, monkeypatch, error):
    from cron import executions, jobs, scheduler

    monkeypatch.setattr(executions, 'EXECUTIONS_FILE', tmp_path / 'executions.db')
    job = jobs.create_job(name='fixture', prompt='fixture', schedule='1h', deliver='none')
    claimed = jobs.claim_job_for_fire(job['id'], return_job=True)
    assert isinstance(claimed, dict)
    eid = executions.create_execution(job['id'], source='direct')['id']
    claimed['execution_id'] = eid
    assert executions.mark_execution_handoff_pending(eid)
    assert executions.adopt_claimed_execution(eid)

    def observe(_job):
        raise scheduler._ExternalWorkerPostHandoffError(str(error)) from error

    monkeypatch.setattr(scheduler, '_launch_external_cron_worker', observe)
    send = Mock()
    monkeypatch.setattr(scheduler, '_deliver_crash_failure', send)
    assert scheduler.run_one_job(claimed)
    assert jobs.get_job(job['id'])['fire_claim'] == claimed['fire_claim']
    assert executions.get_execution(eid)['status'] == 'running'
    assert jobs.claim_job_for_fire(job['id']) is False
    send.assert_not_called()


def test_waiter_busy_read_retries_without_releasing_worker(monkeypatch):
    from cron import scheduler
    import subprocess

    process = Mock()
    process.wait.side_effect = [subprocess.TimeoutExpired('worker', 1), 0]
    rows = iter([sqlite3.OperationalError('database is locked'), {'status': 'completed'}])

    def read(_eid):
        value = next(rows)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(scheduler, 'get_execution', read)
    assert scheduler._wait_for_external_cron_worker_body(process, execution_id='fixture')
    assert process.wait.call_count == 2


@pytest.mark.parametrize('observer_fails',[False,True])
def test_adopted_child_finishes_once_after_parent_observation(tmp_path,monkeypatch,observer_fails):
    """Exercise process-identity fencing, not same-process adoption mocks."""
    import json
    import subprocess
    import sys
    from cron import executions,jobs,scheduler

    ledger=tmp_path/'executions.db'
    monkeypatch.setattr(executions,'EXECUTIONS_FILE',ledger)
    job=jobs.create_job(name='fixture',prompt='fixture',schedule='1h',deliver='none')
    claimed=jobs.claim_job_for_fire(job['id'],return_job=True)
    eid=executions.create_execution(job['id'],source='direct')['id']
    claimed['execution_id']=eid
    assert executions.mark_execution_handoff_pending(eid)
    code='''
import sys,json
from pathlib import Path
from cron import executions,jobs
executions.EXECUTIONS_FILE=Path(sys.argv[1])
job=json.loads(sys.argv[2]); eid=job['execution_id']
assert executions.adopt_claimed_execution(eid)
print('adopted',flush=True)
assert sys.stdin.readline().strip()=='finish'
with jobs.fire_claim_fence(job['id'],expected_owner=job['fire_claim']['by']) as owned:
    assert owned
    with Path(sys.argv[3]).open('a') as effect: effect.write('once\\n')
    assert executions.finish_execution(eid,success=True)
    jobs.mark_job_run(job['id'],True,expected_fire_owner=job['fire_claim']['by'])
'''
    effect=tmp_path/'side-effect'
    child=subprocess.Popen([sys.executable,'-c',code,str(ledger),json.dumps(claimed),str(effect)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='adopted'
        assert executions.get_execution(eid)['pid']==child.pid
        if observer_fails:
            def fail(_job):
                raise scheduler._ExternalWorkerPostHandoffError('database is locked')
            monkeypatch.setattr(scheduler,'_launch_external_cron_worker',fail)
            assert scheduler.run_one_job(claimed)
        assert jobs.get_job(job['id'])['fire_claim']==claimed['fire_claim']
        assert jobs.claim_job_for_fire(job['id']) is False
        _out,err=child.communicate('finish\n',timeout=30)
        assert child.returncode==0,err
        assert executions.get_execution(eid)['status']=='completed'
        assert effect.read_text()=='once\n'
        assert not jobs.get_job(job['id']).get('fire_claim')
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)
