"""``gateway.detached_workers``: a restarted owner keeps the turn of a managed worker that outlived
the previous owner, and only that one. Real SessionDB, real child processes."""
from contextlib import closing
import os
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from hermes_cli.process_identity import worker_birth
from hermes_state import SessionDB
from hermes_state_runtime import (
    RuntimeStoreError, admit_session_input, begin_runtime_epoch, claim_session_input, recover_session_inputs,
)


def _hello(process):
    return {'type': 'hello', 'pid': process.pid, 'birth': worker_birth(psutil.Process(process.pid)),
            'ancestors': [os.getpid()]}


def _reserve(tmp_path, db, *, detach):
    from gateway.session_worker_reservation import reserve_admission_worker
    db.create_session('owned', 'cli')
    epoch = begin_runtime_epoch(db, instance_id='first')
    owner = SimpleNamespace(db=db, epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None)
    admit_session_input(db, epoch=epoch, principal_id='human', session_id='owned', request_id='turn', payload={})
    row = claim_session_input(db, epoch=epoch, session_id='owned')
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    scope = reserve_admission_worker(owner, admission_id=row['admission_id'], process=child, principal_id='human',
                                     hello=_hello(child), worker_subject='uid:1',
                                     detach_socket=str(tmp_path / 'w.sock') if detach else None)
    return row, child, scope


def _restart(tmp_path, db):
    """What the next owner's boot does (``initialize_session_authority``) before any FIFO runs."""
    from gateway.session_detached_workers import spare_live_workers
    epoch = begin_runtime_epoch(db, instance_id='second')
    spared = spare_live_workers(db, epoch)
    recover_session_inputs(db, epoch=epoch)
    return epoch, spared


def _adopt(db, epoch, scope):
    from hermes_state_runtime import adopt_worker_execution
    from gateway.session_admission import admission_fingerprint
    identity = {k: v for k, v in scope.items() if k != 'epoch'}
    claim = admission_fingerprint(canonical_target=scope['session_id'], payload=identity | {'principal': 'uid:1'})
    return adopt_worker_execution(db, epoch=epoch, execution_id=scope['execution_id'], session_id=scope['session_id'],
                                  generation=scope['generation'], adoption_secret=claim)


@pytest.mark.platforms("linux", "macos")
def test_a_live_detached_worker_keeps_its_turn_across_an_owner_restart(tmp_path):
    from hermes_state_runtime import mutate_worker_execution
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        _row, child, scope = _reserve(tmp_path, db, detach=True)
        try:
            epoch, spared = _restart(tmp_path, db)
            assert [r['pid'] for r in spared] == [child.pid]
            assert db._read_one('SELECT status,owner_epoch FROM session_admissions')[:] == ('started', epoch)
            # The worker's own handshake with the new owner now passes, and its finish settles the execution.
            assert _adopt(db, epoch, scope)['owner_epoch'] == epoch
            mutate_worker_execution(db, epoch=epoch, execution_id=scope['execution_id'], session_id=scope['session_id'],
                                    generation=scope['generation'], sequence=1, operation='execution.finish', payload={})
            assert db._read_one('SELECT status FROM worker_executions')[0] == 'terminal'
            assert db._read_one("SELECT COUNT(*) FROM state_meta WHERE key LIKE 'gateway.detached_worker.v1.%'")[0] == 0
        finally:
            child.kill()
            child.wait(timeout=5)


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize('detach', [False, True], ids=['legacy', 'detached-but-dead'])
def test_a_dead_or_undeclared_worker_is_still_unknown_after_a_restart(tmp_path, detach):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        _row, child, scope = _reserve(tmp_path, db, detach=detach)
        if detach:
            child.kill()
            child.wait(timeout=5)
        try:
            epoch, spared = _restart(tmp_path, db)
            assert spared == []
            assert db._read_one('SELECT status FROM session_admissions')[0] == 'unknown'
            with pytest.raises(RuntimeStoreError, match='stale_epoch'):
                _adopt(db, epoch, scope)
            assert db._read_one("SELECT COUNT(*) FROM state_meta WHERE key LIKE 'gateway.detached_worker.v1.%'")[0] == 0
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
