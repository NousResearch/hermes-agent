"""Delivery retries honor the platform deadline without crossing ownership boundaries."""
import os
import sqlite3
import time

from gateway import delivery_ledger as dl


def record(oid, *, profile=None):
    dl.record_obligation(obligation_id=oid, session_key='session-' + oid, platform='telegram',
                         chat_id='123', thread_id='77', content='.' * 3000, adapter_profile=profile)
    dl.mark_failed(oid, 'flood_control:185')


def read(oid):
    with sqlite3.connect(dl._db_path()) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute('SELECT * FROM delivery_obligations WHERE obligation_id=?', (oid,)).fetchone())


def test_runtime_deadline_preserves_scope_and_retry_budget():
    record('due')
    record('other', profile='other')
    record('blocked')
    dl.mark_failed('blocked', 'Forbidden: bot was blocked by the user')
    stamp = read('due')['updated_at']
    assert dl.sweep_failed_for_runtime('telegram', now=stamp + 184) == []
    assert read('due')['attempts'] == 0
    claimed = dl.sweep_failed_for_runtime('telegram', now=stamp + 186)
    assert [r['obligation_id'] for r in claimed] == ['due']
    assert claimed[0]['needs_marker'] and 'rate limit' in claimed[0]['marker']
    assert read('due')['last_error'] is None
    assert dl.sweep_failed_for_runtime('telegram', now=stamp + 187) == []
    assert dl.release_runtime_claim('due', claimed[0]['last_error'])
    assert read('due')['attempts'] == 0
    assert dl.sweep_failed_for_runtime('telegram', now=time.time()) == []
    claimed = dl.sweep_failed_for_runtime('telegram', now=time.time() + 186)
    assert len(claimed) == 1
    dl.mark_delivered('due')
    assert dl.sweep_failed_for_runtime('telegram', now=time.time() + 187) == []
    assert read('other')['attempts'] == read('blocked')['attempts'] == 0


def test_boot_adopts_waiting_rows_without_spending_or_losing_deadline():
    record('boot')
    original = read('boot')
    # No PID means a genuinely ownerless persisted row; do not patch the liveness predicate.
    with sqlite3.connect(dl._db_path()) as conn:
        conn.execute("UPDATE delivery_obligations SET owner_pid=NULL, owner_started_at=NULL, adapter_profile=NULL")
    claimed = dl.sweep_recoverable(now=original['updated_at'] + 20,
                                    deliverable_targets={('telegram', None)})
    assert len(claimed) == 1 and claimed[0].get('adopted')
    adopted = read('boot')
    assert adopted['owner_pid'] == os.getpid()
    assert adopted['adapter_profile'] == 'default'
    assert adopted['attempts'] == 0 and adopted['updated_at'] == original['updated_at']
    assert dl.sweep_recoverable(now=original['updated_at'] + 30) == []
    assert dl.sweep_failed_for_runtime('telegram', now=original['updated_at'] + 184) == []
    claimed = dl.sweep_failed_for_runtime('telegram', now=original['updated_at'] + 186)
    assert len(claimed) == 1 and claimed[0]['needs_marker']
    assert read('boot')['state'] == 'attempting' and read('boot')['last_error'] is None
