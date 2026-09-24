"""Owner-issued admission reservations never weaken idle public registration."""
import os
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from hermes_state import SessionDB
from hermes_state_runtime import (admit_session_input, begin_runtime_epoch, claim_session_input,
                                  register_worker_execution, settle_session_input)


def hello(process):
    """A direct child's self-introduction (no launcher: chain length 0)."""
    return {'type': 'hello', 'pid': process.pid, 'birth': psutil.Process(process.pid).create_time(), 'ancestors': [os.getpid()]}


def test_started_admission_binds_only_exact_owned_process_and_principal(tmp_path):
    from gateway.session_worker_reservation import reserve_admission_worker
    db = SessionDB(tmp_path / 'state.db')
    process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        db.create_session('owned', 'cli')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        authority = SimpleNamespace(db=db, epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None)
        admitted = admit_session_input(db, epoch=epoch, principal_id='human', session_id='owned', request_id='input', payload={})
        kwargs = dict(admission_id=admitted['admission_id'], process=process, principal_id='human', hello=hello(process))
        with pytest.raises(Exception, match='producer_not_started'):
            reserve_admission_worker(authority, **kwargs)
        claim = claim_session_input(db, epoch=epoch, session_id='owned')
        with pytest.raises(Exception, match='permission_denied'):
            reserve_admission_worker(authority, **dict(kwargs, principal_id='other'))
        with pytest.raises(Exception, match='stale_generation'):
            register_worker_execution(db, epoch=epoch, execution_id='public', session_id='owned',
                generation=claim['generation'], kind='compute', adoption_secret='public', require_idle=True)
        scope = reserve_admission_worker(authority, **kwargs)
        assert scope['session_id'] == 'owned' and scope['generation'] == claim['generation']
        assert scope['pid'] == process.pid and scope['profile_id'] == str(tmp_path)
        assert db._read_one('SELECT COUNT(*) FROM worker_executions')[0] == 1
        # The reservation is single-use; another bootstrap cannot replace its secret.
        with pytest.raises(Exception, match='admission_conflict'):
            reserve_admission_worker(authority, **kwargs)
        settle_session_input(db, epoch=epoch, admission_id=admitted['admission_id'], generation=claim['generation'], outcome='completed')
        with pytest.raises(Exception, match='producer_not_started'):
            reserve_admission_worker(authority, **kwargs)
    finally:
        process.stdin.close()
        process.wait(timeout=5)
        db.close()


@pytest.mark.asyncio
async def test_reserved_process_uses_existing_authenticated_adoption_and_persistence(tmp_path):
    from gateway.session_worker import worker_request
    from gateway.session_worker_reservation import reserve_admission_worker
    db = SessionDB(tmp_path / 'state.db')
    process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        db.create_session('owned', 'cli')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        authority = SimpleNamespace(db=db, epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None)
        admitted = admit_session_input(db, epoch=epoch, principal_id='human', session_id='owned', request_id='input', payload={})
        claim_session_input(db, epoch=epoch, session_id='owned')
        scope = reserve_admission_worker(authority, admission_id=admitted['admission_id'], process=process, principal_id='human', hello=hello(process))
        actor = SimpleNamespace(subject='human', profile_id=str(tmp_path), capabilities={'worker:adopt'})
        connection = SimpleNamespace(authority=authority, actor=actor)
        identity = {k: v for k, v in scope.items() if k != 'epoch'}
        with pytest.raises(Exception, match='permission_denied'):
            await worker_request(connection, None, identity | {'secret': 'wrong-secret-for-reservation'}, operation='adopt')
        receipt = await worker_request(connection, None, identity, operation='adopt')
        assert receipt['execution_id'] == scope['execution_id']
        params = scope | {'sequence': 1, 'operation': 'execution.finish', 'payload': {}}
        result = await worker_request(connection, None, params, operation='persist')
        assert result['status'] == 'terminal'
        with pytest.raises(Exception, match='stale_generation'):
            await worker_request(connection, None, params | {'sequence': 2}, operation='persist')
    finally:
        process.stdin.close()
        process.wait(timeout=5)
        db.close()
