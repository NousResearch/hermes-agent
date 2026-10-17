"""Worker compression retains local guard and transcript contracts."""
import sqlite3
import time

import pytest

from hermes_state_runtime import mutate_worker_execution


def test_guard_rollback_and_streaks_are_durable_receipts(worker):
    db, store = worker
    deadline = time.time() + 3600
    snapshot = store.get_compression_failure_cooldown_row('owned')
    store.record_compression_failure_cooldown('owned', deadline, 'first')
    store.record_compression_failure_cooldown('owned', deadline - 100, 'latest')
    assert db.get_compression_failure_cooldown_row('owned') == {
        'session_exists': True, 'cooldown_until': deadline, 'error': 'latest'}
    store.restore_compression_failure_cooldown_row('owned', snapshot)
    assert db.get_compression_failure_cooldown_row('owned') == snapshot
    store.record_compression_failure_cooldown('owned', deadline)
    store.clear_compression_failure_cooldown('owned')
    assert store.get_compression_failure_cooldown('owned') is None
    store.set_compression_fallback_streak('owned', 4)
    store.set_compression_ineffective_count('owned', 3)
    store.set_compression_recovery_deadline('owned', deadline)
    assert (db.get_compression_fallback_streak('owned'), db.get_compression_ineffective_count('owned'),
            db.get_compression_recovery_deadline('owned')) == (4, 3, deadline)
    seq = store.journal['next_sequence']
    with pytest.raises(Exception, match='cannot restore absent'):
        mutate_worker_execution(db, **store.scope, sequence=seq, operation='compression.cooldown.restore',
                                payload={'snapshot': {'session_exists': False, 'cooldown_until': None, 'error': None}})
    assert db._read_one('SELECT last_sequence FROM worker_executions')[0] == seq - 1


def test_compression_lease_cannot_be_revived_or_released_by_old_holder(worker):
    db, store = worker
    assert store.try_acquire_compression_lock('owned', 'old')
    assert not store.try_acquire_compression_lock('owned', 'new')
    db._write_sql('UPDATE compression_locks SET expires_at=0 WHERE session_id=?', ('owned',))
    assert store.refresh_compression_lock('owned', 'old')
    db._write_sql('UPDATE compression_locks SET expires_at=0 WHERE session_id=?', ('owned',))
    assert store.try_acquire_compression_lock('owned', 'new')
    assert not store.refresh_compression_lock('owned', 'old')
    store.release_compression_lock('owned', 'old')
    assert store.get_compression_lock_holder('owned') == 'new'
    seq = store.journal['next_sequence']
    for scope, error in [(dict(store.scope, epoch=store.scope['epoch'] - 1), 'stale_epoch'),
                         (dict(store.scope, session_id='foreign'), 'permission_denied')]:
        with pytest.raises(Exception, match=error):
            mutate_worker_execution(db, **scope, sequence=seq, operation='compression.lock.release', payload={'holder': 'new'})
    assert db.get_compression_lock_holder('owned') == 'new'
    store.release_compression_lock('owned', 'new')
    assert store.get_compression_lock_holder('owned') is None


@pytest.mark.parametrize('exact_coverage', [False, True])
@pytest.mark.parametrize('prompt', ['NEW PROMPT', ''])
@pytest.mark.parametrize('fail_prompt', [False, True])
def test_worker_compaction_commits_prompt_and_transcript_together(worker, exact_coverage, prompt, fail_prompt):
    db, store = worker
    original = [{'role': 'user', 'content': 'original question'},
                {'role': 'assistant', 'content': 'original answer'}]
    store.append_messages_batch('owned', original)
    covered = db.get_active_message_ids('owned') if exact_coverage else None
    compacted = [{'role': 'user', 'content': 'summary'}]
    sequence = store.journal['next_sequence']
    if fail_prompt:
        db._conn.execute("CREATE TEMP TRIGGER reject_worker_prompt BEFORE UPDATE OF system_prompt_hash ON sessions "
                         "WHEN NEW.id = 'owned' BEGIN SELECT RAISE(FAIL, 'worker prompt refused'); END")
        with pytest.raises(sqlite3.IntegrityError, match='worker prompt refused'):
            store.archive_and_compact('owned', compacted, covered_ids=covered, system_prompt=prompt)
        assert [row['content'] for row in db.get_messages('owned')] == ['original question', 'original answer']
        assert db.get_session('owned')['system_prompt'] == 'prefix'
        assert db._read_one('SELECT last_sequence FROM worker_executions')[0] == sequence - 1
        db._conn.execute('DROP TRIGGER reject_worker_prompt')
        store.retry_pending()
    else:
        assert store.archive_and_compact('owned', compacted, covered_ids=covered, system_prompt=prompt) == 1
    assert [row['content'] for row in db.get_messages('owned')] == ['summary']
    assert db.get_session('owned')['system_prompt'] == prompt
    assert db.get_session('foreign')['system_prompt'] == 'secret'
    assert db._read_one('SELECT last_sequence FROM worker_executions')[0] == sequence


def test_worker_compaction_rejects_non_text_prompt_before_mutation(worker):
    from hermes_state_runtime import RuntimeStoreError

    db, store = worker
    sequence = store.journal['next_sequence']
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        mutate_worker_execution(db, **store.scope, sequence=sequence, operation='compression.archive',
            payload=dict(messages=[{'role': 'user', 'content': 'must not land'}], model_config_patch=None,
                         watermark=None, lock_holder=None, tail_count=0, system_prompt=123))
    assert db.get_messages('owned') == []
    assert db.get_session('owned')['system_prompt'] == 'prefix'
    assert db._read_one('SELECT last_sequence FROM worker_executions')[0] == sequence - 1
