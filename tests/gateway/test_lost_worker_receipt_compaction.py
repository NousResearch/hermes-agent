"""A managed worker closed by loss or by a reset's orphan discard keeps no full-history read
receipt (dokterdok N41, the N19 residual): every terminal path compacts like execution.finish."""
from contextlib import closing
import os
import subprocess
import sys
from types import SimpleNamespace

import psutil

from hermes_state import SessionDB
from hermes_state_runtime import (
    admit_session_input, begin_runtime_epoch, claim_session_input, mutate_worker_execution,
    recover_session_inputs, register_worker_execution,
)

READ = dict(target='s', include_ancestors=False, include_inactive=False, repair_alternation=False,
            include_row_ids=False, include_compacted=False)


def hello(process):
    return {'type': 'hello', 'pid': process.pid, 'birth': psutil.Process(process.pid).create_time(), 'ancestors': [os.getpid()]}


def _receipt_sizes(db):
    return [n for (n,) in db._read_all('SELECT LENGTH(result_json) FROM worker_receipts ORDER BY sequence')]


def test_lost_admission_worker_compacts_its_history_receipt(tmp_path):
    from gateway.session_worker_reservation import lose_admission_worker, reserve_admission_worker
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        db.create_session('s', 'cli')
        db.append_messages_batch('s', [{'role': 'user', 'content': 'x' * 4000}])
        epoch = begin_runtime_epoch(db, instance_id='owner')
        authority = SimpleNamespace(db=db, epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None)
        admit_session_input(db, epoch=epoch, principal_id='human', session_id='s', request_id='input', payload={})
        row = claim_session_input(db, epoch=epoch, session_id='s')
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            scope = reserve_admission_worker(authority, admission_id=row['admission_id'], process=child,
                                             principal_id='human', hello=hello(child))
            history = mutate_worker_execution(db, epoch=epoch, execution_id=scope['execution_id'], session_id='s',
                generation=scope['generation'], sequence=1, operation='compression.history', payload=READ)
            assert history['messages'][0]['content'] == 'x' * 4000
            lose_admission_worker(authority, row, scope)
            assert db._read_one('SELECT status FROM worker_executions')[0] == 'terminal'
            assert max(_receipt_sizes(db)) < 200, 'a lost worker kept a full-history receipt'
        finally:
            child.stdin.close()
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def test_reset_discarded_orphan_worker_compacts_its_history_receipt(tmp_path):
    from hermes_state_runtime_workers import discard_orphan_workers_on_reset
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        db.create_session('s', 'cli')
        db.append_messages_batch('s', [{'role': 'user', 'content': 'x' * 4000}])
        epoch = begin_runtime_epoch(db, instance_id='old-owner')
        scope = dict(epoch=epoch, execution_id='orphan', session_id='s', generation=0)
        register_worker_execution(db, **scope, kind='compute', adoption_secret='secret')
        mutate_worker_execution(db, **scope, sequence=1, operation='compression.history', payload=READ)
        recover_session_inputs(db, epoch=begin_runtime_epoch(db, instance_id='new-owner'))  # prior owner's compute -> unknown
        assert db._read_one('SELECT status FROM worker_executions')[0] == 'unknown'
        db._execute_write(lambda conn: discard_orphan_workers_on_reset(conn, ['s']))
        assert db._read_one('SELECT status FROM worker_executions')[0] == 'terminal'
        assert max(_receipt_sizes(db)) < 200, 'a reset-discarded worker kept a full-history receipt'
