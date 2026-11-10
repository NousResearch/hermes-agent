"""Exact receipt reads preserve fences and never advance the writer stream."""
import sqlite3
import time
import pytest
from hermes_state import SessionDB
from hermes_state_runtime import (RuntimeStoreError, begin_runtime_epoch,
    register_worker_execution, mutate_worker_execution)


@pytest.fixture
def worker(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        db.create_session('assigned', source='cli')
        db.create_session('foreign', source='cli')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        register_worker_execution(db, epoch=epoch, execution_id='worker', session_id='assigned',
                                  generation=0, kind='compute', adoption_secret='synthetic')
        yield db, dict(epoch=epoch, execution_id='worker', session_id='assigned', generation=0,
                       sequence=1, operation='usage.main', payload={'input_tokens': 7})


def test_read_exact_receipt_without_write_lock_or_duplicate_effect(worker, caplog):
    from hermes_state_worker_receipts import read_worker_receipt
    db, params = worker
    caplog.set_level(10, logger="hermes_state")
    before = time.time()
    result = mutate_worker_execution(db, **params)
    stages = [record.getMessage() for record in caplog.records if 'state_write stage=' in record.getMessage()]
    assert any('stage=lockwait' in message for message in stages)
    assert any('stage=commit' in message for message in stages)
    assert all('assigned' not in message and 'worker' not in message and 'input_tokens' not in message for message in stages)
    created = db._read_one('SELECT created_at FROM worker_receipts')[0]
    assert before <= created <= time.time()
    other = sqlite3.connect(db.db_path, isolation_level=None)
    try:
        other.execute('BEGIN IMMEDIATE')
        assert read_worker_receipt(db, **params) == result
        assert db.get_session('assigned')['input_tokens'] == 7
        assert db._read_one('SELECT last_sequence FROM worker_executions')[0] == 1
    finally:
        other.rollback()
        other.close()


@pytest.mark.parametrize('change,reason', [
    ({'epoch': 0}, 'stale_epoch'), ({'generation': 2}, 'stale_generation'),
    ({'session_id': 'foreign'}, 'permission_denied'),
    ({'payload': {'input_tokens': 8}}, 'admission_conflict'),
    ({'sequence': 2}, 'not_found'), ({'sequence': True}, 'invalid_params')])
def test_reconcile_fails_closed(worker, change, reason):
    from hermes_state_worker_receipts import read_worker_receipt
    db, params = worker
    mutate_worker_execution(db, **params)
    with pytest.raises(RuntimeStoreError, match=reason):
        read_worker_receipt(db, **(params | change))
    assert db._read_one('SELECT COUNT(*) FROM worker_receipts')[0] == 1


def test_retired_history_receipt_is_not_a_successful_reconciliation(worker):
    from hermes_state_worker_receipts import read_worker_receipt
    db, params = worker
    params = params | {'operation': 'compression.history', 'payload': dict(target='assigned',
        include_ancestors=False, include_inactive=False, repair_alternation=False,
        include_row_ids=False, include_compacted=False)}
    mutate_worker_execution(db, **params)
    mutate_worker_execution(db, **(params | {'sequence': 2, 'operation': 'execution.finish', 'payload': {}}))
    with pytest.raises(RuntimeStoreError, match='stale_generation'):
        read_worker_receipt(db, **params)


def test_legacy_timestamp_stays_unknown_and_new_receipts_are_stamped(worker):
    db, params = worker
    mutate_worker_execution(db, **params)
    path = db.db_path
    db.close()
    with sqlite3.connect(path) as conn:
        conn.execute('ALTER TABLE worker_receipts DROP COLUMN created_at')
    with SessionDB(path) as reopened:
        assert reopened._read_one('SELECT created_at FROM worker_receipts')[0] is None
        mutate_worker_execution(reopened, **(params | {'sequence': 2}))
        stamps = reopened._read_all('SELECT created_at FROM worker_receipts ORDER BY sequence')
        assert stamps[0][0] is None and stamps[1][0] > 0
