"""Cancelled attempts compact only after their authoritative epoch is superseded."""
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore


def test_many_cancelled_attempts_compact_across_authority_epochs(tmp_path):
    path = tmp_path / 'runs.db'
    old = []
    page_counts = []
    for epoch in range(1, 7):
        store = RunIdempotencyStore(str(path))
        try:
            scope, authority = f'scope-{epoch}', ('room-member-target', epoch, 'home')
            assert store.observe_room_authority(scope, authority)
            # A thousand absent identities and a thousand stopped terminal runs
            # per epoch reproduce both immortal-row classes from the review.
            for index in range(1000):
                key, run_id = f'absent-{index}', f'absent-{epoch}-{index}'
                store.reserve(scope, key, '', run_id,
                              {'status': 'cancelled', 'admission_cancelled': True},
                              cancel_if_missing=True, room_authority=authority)
                old.append((scope, key, authority))
                key, run_id = f'stopped-{index}', f'stopped-{epoch}-{index}'
                store.reserve(scope, key, 'fingerprint', run_id, {'status': 'completed'},
                              room_authority=authority)
                store.request_stop(scope, run_id)
                old.append((scope, key, authority))
            store._conn.execute('UPDATE run_idempotency SET updated_at=0,retention_until=1')
            store._conn.commit()
            # A fresh grant for the SAME epoch cannot forget an exact cancellation.
            store.observe_room_authority(scope, authority)
            assert store.lookup(scope, 'absent-0', 'changed', room_authority=authority)[0] == 'reused'
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 2000
            # The authenticated successor retires the old execution namespace.
            store.observe_room_authority(f'scope-{epoch + 1}', (authority[0], epoch + 1, 'home'))
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 0
            assert store._conn.execute('SELECT COUNT(*) FROM run_room_authorities').fetchone()[0] == 1
            store._conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            page_counts.append(store._conn.execute('PRAGMA page_count').fetchone()[0])
        finally:
            store.close()
    # SQLite reuses the bounded peak allocation, including across process-style reopen.
    assert max(page_counts) <= page_counts[0] * 1.1
    store = RunIdempotencyStore(str(path))
    try:
        for scope, key, authority in old:
            outcome, _ = store.reserve(scope, key, 'late', 'never-start', {'status': 'queued'},
                                       room_authority=authority)
            assert outcome == 'authority_retired'
        assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 0
    finally:
        store.close()


def test_retirement_preserves_live_stop_until_owner_settles(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        authority = ('room-member-target', 1, 'home')
        store.observe_room_authority('first', authority)
        store.reserve('first', 'running', 'f', 'live', {'status': 'running'}, room_authority=authority)
        store.request_stop('first', 'live')
        store.observe_room_authority('next', (authority[0], 2, 'successor'))
        assert store.stop_requested('live')
        assert store.status_for_run('first', 'live')['status']['status'] == 'running'
        store.update_status('live', {'status': 'cancelled'})
        assert store.status_for_run('first', 'live')['status']['status'] == 'cancelled'
        store._conn.execute('UPDATE run_idempotency SET updated_at=0,retention_until=1')
        store._conn.commit()
        store.observe_room_authority('next', (authority[0], 2, 'successor'))
        assert store.status_for_run('first', 'live') is None
        assert store.reserve('first', 'running', 'f', 'late', {'status': 'queued'},
                             room_authority=authority)[0] == 'authority_retired'
    finally:
        store.close()


def test_retirement_keeps_unclassifiable_execution_evidence(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        authority = ('room-member-target', 1, 'home')
        store.observe_room_authority('first', authority)
        store.reserve('first', 'unknown', 'f', 'unknown-run', {'status': 'running'}, room_authority=authority)
        store.request_stop('first', 'unknown-run')
        store._conn.execute("UPDATE run_idempotency SET status_json='[]'")
        store._conn.commit()
        store.retire_room_authority('first', authority)
        assert store._conn.execute('SELECT status_json,stop_requested FROM run_idempotency').fetchall() == [('[]', 1)]
        assert store.reserve('first', 'another', 'f', 'late', {'status': 'queued'},
                             room_authority=authority) == ('authority_retired', None)
    finally:
        store.close()
