"""Compression cleanup keeps physical and logical lease identities separate."""
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session
from tests.persistence_sqlite.workers._runtime_worker import worker as worker  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import


def test_rotation_cleanup_releases_only_its_original_compression_holder(worker):
    persistence, store = worker
    assert store.try_acquire_compression_lock('owned', 'holder')
    store.publish_compression_child(parent_session_id='owned', child_session_id='child', source='cli',
        messages=[{'role': 'assistant', 'content': 'summary'}], compression_lock_holder='holder')
    store.release_compression_lock('owned', 'wrong')
    assert persistence.coordination.get_compression_lock_holder(session_id='owned') == 'holder'
    store.release_compression_lock('owned', 'holder')
    assert persistence.coordination.get_compression_lock_holder(session_id='owned') is None


def test_reopen_orphaned_parent_does_not_steal_live_or_published_continuations(worker):
    persistence, store = worker
    holder = persistence.holder()
    end_session(holder, 'owned', 'compression')
    assert store.try_acquire_compression_lock('owned', 'holder')
    assert not store.reopen_orphaned_compression_session('owned')
    holder._write_sql('UPDATE compression_locks SET expires_at=0 WHERE session_id=?', ('owned',))
    assert store.reopen_orphaned_compression_session('owned')
    assert get_session(holder, 'owned')['ended_at'] is None
    assert not store.refresh_compression_lock('owned', 'holder')
    end_session(holder, 'owned', 'compression')
    create_session(holder, 'published', 'cli', parent_session_id='owned')
    assert not store.reopen_orphaned_compression_session('owned')
    assert get_session(holder, 'owned')['end_reason'] == 'compression'
