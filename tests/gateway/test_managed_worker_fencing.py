"""Loss is not success; delayed parent results cannot revive an admission."""
from contextlib import closing
import os
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from gateway.session_authority import admission_row
from hermes_cli.process_identity import worker_birth
from persistence.core.errors import AuthorityRefusal
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def hello(process):
    """A direct child's self-introduction (no launcher: chain length 0)."""
    return {'type': 'hello', 'pid': process.pid, 'birth': worker_birth(psutil.Process(process.pid)), 'ancestors': [os.getpid()]}


def test_worker_loss_fences_same_generation_parent_result(tmp_path):
    from gateway.session_worker_reservation import reserve_admission_worker, lose_admission_worker
    from gateway.session_results import retain_result
    with closing(open_persistence(profile_scope(home=tmp_path))) as db:
        seed_session(db, 'owned', 'cli')
        epoch = db.runtime.begin_epoch(instance_id='owner')
        authority = SimpleNamespace(epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None,
                                    persistence=storage_handle(tmp_path))
        db.admissions.admit(epoch=epoch, principal_id='human', session_id='owned', request_id='input', payload={})
        row = admission_row(db.admissions.claim(epoch=epoch, session_id='owned'))
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            scope = reserve_admission_worker(authority, admission_id=row['admission_id'], process=child, principal_id='human', hello=hello(child))
            child.kill()
            child.wait(timeout=5)
            lose_admission_worker(authority, row, scope)
            with pytest.raises(Exception, match='stale_generation'):
                retain_result(storage_handle(tmp_path), epoch=epoch, row=row, result={'result': {'final_response': 'STALE'}, 'usage': {}})
            with raw_holder(db)._read_ctx() as conn:
                assert conn.execute('SELECT status FROM session_admissions').fetchone()[0] == 'unknown'
                assert conn.execute('SELECT status FROM worker_executions').fetchone()[0] == 'terminal'
        finally:
            child.stdin.close()
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


@pytest.mark.asyncio
async def test_late_bootstrap_handshake_cannot_rearm_a_recovered_admission(tmp_path):
    """The owner reserved and bootstrapped a managed child, then died before the child's first
    ``worker.adopt``; that handshake reaches the restarted owner. Adopting it would flip the
    recovered ``unknown`` admission back to ``started`` under an owner with no pipe to the child
    (which refuses the foreign epoch and exits), stranding the FIFO where Discard cannot reach it."""
    from gateway.session_worker import worker_request
    from gateway.session_worker_reservation import reserve_admission_worker
    with closing(open_persistence(profile_scope(home=tmp_path))) as db:
        seed_session(db, 'owned', 'cli')
        epoch = db.runtime.begin_epoch(instance_id='owner')
        authority = SimpleNamespace(epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None,
                                    persistence=storage_handle(tmp_path))
        lost, _ = db.admissions.admit(epoch=epoch, principal_id='human', session_id='owned', request_id='lost', payload={})
        row = admission_row(db.admissions.claim(epoch=epoch, session_id='owned'))
        follower, _ = db.admissions.admit(epoch=epoch, principal_id='human', session_id='owned', request_id='next', payload={})
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],  # noqa: ASYNC220 -- Popen returns immediately; the test drives the child through its handle
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            scope = reserve_admission_worker(authority, admission_id=row['admission_id'], process=child,
                                             principal_id='human', hello=hello(child))
            restarted = db.runtime.begin_epoch(instance_id='restarted')
            db.admissions.recover(epoch=restarted)
            owner = SimpleNamespace(epoch=restarted, profile_id=str(tmp_path), _require_admission_open=lambda: None,
                                    persistence=storage_handle(tmp_path))
            actor = SimpleNamespace(subject='human', profile_id=str(tmp_path), capabilities={'worker:adopt'})
            identity = {k: v for k, v in scope.items() if k != 'epoch'}
            with pytest.raises(AuthorityRefusal, match='stale_epoch'):
                await worker_request(SimpleNamespace(authority=owner, actor=actor), None, identity, operation='adopt')
            lost = db.admissions.get(admission_id=lost.admission_id)
            assert (lost.status, lost.owner_epoch) == ('unknown', epoch)
            with raw_holder(db)._read_ctx() as conn:
                assert conn.execute('SELECT status,owner_epoch FROM worker_executions').fetchone()[:] == ('unknown', epoch)
            db.admissions.resolve_unknown(epoch=restarted, admission_id=lost.admission_id, generation=row['generation'])
            assert db.admissions.claim(epoch=restarted, session_id='owned').admission_id == follower.admission_id
        finally:
            child.stdin.close()
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
