"""Compression cleanup keeps physical and logical lease identities separate."""
from tests.state.test_runtime_worker_compression import worker  # noqa: F401


def test_rotation_cleanup_releases_only_its_original_compression_holder(worker):
    db, store = worker
    assert store.try_acquire_compression_lock('owned', 'holder')
    store.publish_compression_child(parent_session_id='owned', child_session_id='child', source='cli',
        messages=[{'role': 'assistant', 'content': 'summary'}], compression_lock_holder='holder')
    store.release_compression_lock('owned', 'wrong')
    assert db.get_compression_lock_holder('owned') == 'holder'
    store.release_compression_lock('owned', 'holder')
    assert db.get_compression_lock_holder('owned') is None


def test_reopen_orphaned_parent_does_not_steal_live_or_published_continuations(worker):
    db, store = worker
    db.end_session('owned', 'compression')
    assert store.try_acquire_compression_lock('owned', 'holder')
    assert not store.reopen_orphaned_compression_session('owned')
    db._write_sql('UPDATE compression_locks SET expires_at=0 WHERE session_id=?', ('owned',))
    assert store.reopen_orphaned_compression_session('owned')
    assert db.get_session('owned')['ended_at'] is None
    assert not store.refresh_compression_lock('owned', 'holder')
    db.end_session('owned', 'compression')
    db.create_session('published', 'cli', parent_session_id='owned')
    assert not store.reopen_orphaned_compression_session('owned')
    assert db.get_session('owned')['end_reason'] == 'compression'
