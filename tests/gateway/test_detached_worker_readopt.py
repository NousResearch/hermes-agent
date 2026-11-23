"""SQLite transfers both owners or neither, with an actual disposable worker identity."""

import sqlite3
import subprocess
import sys
from contextlib import closing

import psutil
import pytest

from hermes_cli.process_identity import worker_birth
from persistence.core.errors import AuthorityRefusal
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.fixture
def reserved(tmp_path):
    store = storage_handle(tmp_path)
    seed_session(store, 's', 'gui')
    epoch = store.runtime.begin_epoch(instance_id='old')
    admitted, _ = store.admissions.admit(epoch=epoch, principal_id='fixture', session_id='s',
                                         request_id='first', payload={'text': 'fixture'})
    claimed = store.admissions.claim(epoch=epoch, session_id='s')
    assert claimed.admission_id == admitted.admission_id
    with subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.read()'],
                          stdin=subprocess.PIPE) as child:
        try:
            yield store, epoch, claimed, child
        finally:
            child.stdin.close()
            child.wait(timeout=5)


def _reserve(store, old, claimed, child, *, birth_offset=0):
    return store.worker_executions.reserve_for_admission(
        epoch=old, admission_id=claimed.admission_id, principal_id='fixture', profile_id='fixture',
        pid=child.pid, birth=worker_birth(psutil.Process(child.pid)) + birth_offset,
        secret='unused-by-this-fixture', owner=None, detach_socket='/fixture/worker.sock',
    )


def _worker_epoch(home, execution_id):
    with closing(sqlite3.connect(home / 'state.db')) as connection:
        return connection.execute('SELECT owner_epoch FROM worker_executions WHERE execution_id=?',
                                  (execution_id,)).fetchone()[0]


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('identity', ['live', 'exited', 'reused-pid'])
def test_recover_spares_only_a_live_matching_process(reserved, identity, tmp_path):
    store, old, claimed, child = reserved
    scope = _reserve(store, old, claimed, child, birth_offset=(100 if identity == 'reused-pid' else 0))
    if identity == 'exited':
        child.stdin.close()
        child.wait(timeout=5)
    new = store.runtime.begin_epoch(instance_id='new')
    store.admissions.recover(epoch=new)
    store.admissions.recover(epoch=new)  # a repeat must not undo an already transferred row
    row = store.admissions.get(admission_id=claimed.admission_id)
    records = store.worker_executions.detached_reservations(epoch=new)
    worker_epoch = _worker_epoch(tmp_path, scope.execution_id)
    if identity == 'live':
        assert (row.status, row.owner_epoch) == ('started', new)
        assert worker_epoch == new
        assert len(records) == 1
        assert records[0]['execution_id'] == scope.execution_id
    else:
        assert row.status == 'unknown'
        assert worker_epoch == old
        assert records == ()


@pytest.mark.platforms('linux')
def test_recover_rolls_back_worker_update_if_admission_cannot_move(reserved, tmp_path):
    store, old, claimed, child = reserved
    scope = _reserve(store, old, claimed, child)
    new = store.runtime.begin_epoch(instance_id='new')
    # Simulate an UPDATE which affects zero admission rows, after the worker UPDATE succeeded.
    with closing(sqlite3.connect(tmp_path / 'state.db')) as connection:
        connection.execute('''CREATE TRIGGER reject_transfer BEFORE UPDATE OF owner_epoch ON session_admissions
                              BEGIN SELECT RAISE(IGNORE); END''')
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        store.admissions.recover(epoch=new)
    row = store.admissions.get(admission_id=claimed.admission_id)
    assert (row.status, row.owner_epoch, _worker_epoch(tmp_path, scope.execution_id)) == ('started', old, old)
    with closing(sqlite3.connect(tmp_path / 'state.db')) as connection:
        connection.execute('DROP TRIGGER reject_transfer')
    store.admissions.recover(epoch=new)
    assert store.admissions.get(admission_id=claimed.admission_id).owner_epoch == new
    assert _worker_epoch(tmp_path, scope.execution_id) == new
