"""Maintenance I/O is not part of a committed write's acknowledgement."""
import threading
from concurrent.futures import ThreadPoolExecutor
from hermes_state import SessionDB


def test_blocked_maintenance_does_not_delay_committed_write_and_close_drains(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('assigned', source='cli')
    entered, release, merged, closing = (threading.Event() for _ in range(4))
    calls = []
    def checkpoint():
        calls.append('checkpoint')
        entered.set()
        assert release.wait(10)
    def merge():
        assert db._conn is not None
        calls.append('merge')
        merged.set()
    monkeypatch.setattr(db, '_try_wal_checkpoint', checkpoint)
    monkeypatch.setattr(db, '_try_incremental_merge_fts', merge)
    monkeypatch.setattr(db, '_CHECKPOINT_EVERY_N_WRITES', 1)
    monkeypatch.setattr(db, '_FTS_MERGE_EVERY_N_WRITES', 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            write = pool.submit(db._write_sql, 'UPDATE sessions SET title=? WHERE id=?', ('first', 'assigned'))
            assert entered.wait(10)
            write.result(timeout=2)  # base blocks here until the checkpoint returns
            assert db.get_session('assigned')
            for n in range(10):
                db._write_sql('UPDATE sessions SET title=? WHERE id=?', (str(n), 'assigned'))
            def close():
                closing.set()
                db.close()
            closed = pool.submit(close)
            assert closing.wait(2)
            assert db._conn is not None  # close must not retire underneath maintenance
        finally:
            release.set()
        closed.result(timeout=10)
    assert merged.is_set() and db._conn is None
    assert calls.count('checkpoint') == 2  # active + coalesced pending, not ten queued jobs
    assert calls.count('merge') == 2


def test_reopened_writer_gets_a_new_maintenance_lifecycle(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('assigned', source='cli')
    db.close()
    checkpointed = threading.Event()
    monkeypatch.setattr(db, '_CHECKPOINT_EVERY_N_WRITES', 1)
    monkeypatch.setattr(db, '_try_wal_checkpoint', checkpointed.set)
    try:
        db._write_sql('UPDATE sessions SET title=? WHERE id=?', ('late', 'assigned'))
        assert checkpointed.wait(10)
    finally:
        db.close()


def test_thread_start_failure_keeps_maintenance_due_until_close(tmp_path, monkeypatch):
    import hermes_state_write_maintenance as maintenance
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('assigned', source='cli')
    calls = []
    def cannot_spawn(*args, **kwargs):
        raise RuntimeError('synthetic start failure')
    monkeypatch.setattr(maintenance, 'spawn_context_thread', cannot_spawn)
    monkeypatch.setattr(db, '_CHECKPOINT_EVERY_N_WRITES', 1)
    monkeypatch.setattr(db, '_FTS_MERGE_EVERY_N_WRITES', 1)
    monkeypatch.setattr(db, '_try_wal_checkpoint', lambda: calls.append('checkpoint'))
    monkeypatch.setattr(db, '_try_incremental_merge_fts', lambda: calls.append('merge'))
    db._write_sql('UPDATE sessions SET title=? WHERE id=?', ('committed', 'assigned'))
    assert db.get_session('assigned')['title'] == 'committed'
    db.close()
    assert calls == ['checkpoint', 'merge']
