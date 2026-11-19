"""Display reads hide a message whose newest copy was removed (rewound, or replaced by a newer copy), so
they need the removed rows' identities, but never their bodies: ``assert_resume_safe`` does not count
those rows, and repeated in-place compaction leaves one removed copy of the carried tail per round
(review of #140971). The SQLite backend's display reads on a real StoreHolder, real ``archive_and_compact``."""
import sqlite3

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence.sqlite.conversation._store.messages import (
    append_messages_batch,
    archive_and_compact,
    get_messages,
    get_messages_as_conversation,
    get_resume_conversations,
    rewind_to_message,
)
from persistence.sqlite.conversation._store.sessions import create_session

BIG = 'x' * 100_000


def _compacted_repeatedly(db, rounds):
    create_session(db, 's', source='test')
    append_messages_batch(db, 's', [{'role': 'user', 'content': 'U1'}, {'role': 'assistant', 'content': 'A1'},
                                    {'role': 'user', 'content': 'U2'}, {'role': 'assistant', 'content': BIG}])
    for n in range(rounds):
        # /compress with nothing new since: the same verbatim tail (U2 + the big reply) is carried again.
        archive_and_compact(db, 's', [{'role': 'user', 'content': f'[CONTEXT SUMMARY] round {n}'},
                                      {'role': 'user', 'content': 'U2'}, {'role': 'assistant', 'content': BIG}],
                            tail_count=2)


def _big_bodies_read(db, monkeypatch, read):
    seen = []
    real = db._read_all

    def spy(sql, params=()):
        rows = real(sql, params)
        seen.extend(r['content'] for r in rows
                    if isinstance(r, sqlite3.Row) and 'content' in r.keys() and r['content'] == BIG)
        return rows
    monkeypatch.setattr(db, '_read_all', spy)
    shown = read()
    monkeypatch.undo()
    return shown, len(seen)


def test_display_reads_fetch_no_body_of_a_removed_copy(tmp_path, monkeypatch):
    with StoreHolder(tmp_path / 'state.db') as db:
        _compacted_repeatedly(db, 6)
        removed = db._read_one("SELECT COUNT(*) FROM messages WHERE content = ? AND active = 0 AND compacted = 0",
                               (BIG,))[0]
        assert removed >= 5  # the fixture really leaves removed copies behind
        archived = db._read_one("SELECT COUNT(*) FROM messages WHERE content = ? AND compacted = 1", (BIG,))[0]
        for read in (
            lambda: get_messages_as_conversation(db, 's', include_ancestors=True, include_compacted=True),
            lambda: get_resume_conversations(db, 's')[1],
        ):
            shown, bodies = _big_bodies_read(db, monkeypatch, read)
            assert sum(1 for m in shown if m.get('content') == BIG) == 1
            # Only the live copy and the compaction-archived ones (what the resume guard counts) carry a body.
            assert bodies <= 1 + archived, (bodies, archived, removed)


def test_a_rewound_turn_stays_hidden_without_its_body(tmp_path):
    with StoreHolder(tmp_path / 'state.db') as db:
        create_session(db, 's', source='test')
        append_messages_batch(db, 's', [{'role': 'user', 'content': 'KEEP'}, {'role': 'assistant', 'content': 'OK'},
                                        {'role': 'user', 'content': 'GONE'}, {'role': 'assistant', 'content': BIG}])
        rewind_to_message(db, 's', next(m['id'] for m in get_messages(db, 's') if m['content'] == 'GONE'))
        shown = get_messages_as_conversation(db, 's', include_ancestors=True, include_compacted=True)
        assert [m['content'] for m in shown] == ['KEEP', 'OK']


def test_the_worker_history_read_agrees_with_the_owner_display_read(backend, scope_a):
    """``worker_history`` (a worker's display read) applies the same removed-copy rule as the owner's."""
    from persistence.sqlite.workers._compression import worker_history
    store = backend.open(scope_a)
    db = store.holder()
    create_session(db, 's', source='test')
    append_messages_batch(db, 's', [{'role': 'user', 'content': 'KEEP'}, {'role': 'assistant', 'content': 'OK'},
                                    {'role': 'user', 'content': 'GONE'}, {'role': 'assistant', 'content': 'NO'}])
    # Rotated: the parent keeps its copies active, the child's copies of GONE/NO are rewound.
    assert store.coordination.try_acquire_compression_lock(session_id='s', holder='fixture')
    try:
        publish_compression_child(db, parent_session_id='s', child_session_id='c', source='test', messages=[
            {'role': 'user', 'content': 'SUMMARY'}, {'role': 'user', 'content': 'GONE'},
            {'role': 'assistant', 'content': 'NO'}], compression_lock_holder='fixture')
    finally:
        store.coordination.release_compression_lock(session_id='s', holder='fixture')
    rewind_to_message(db, 'c', next(m['id'] for m in get_messages(db, 'c') if m['content'] == 'GONE'))
    owner = [m['content'] for m in get_messages_as_conversation(db, 'c', include_ancestors=True,
                                                                include_compacted=True)]
    with db._read_ctx() as conn:
        worker = worker_history(db, conn, 'c', {
            'target': 'c', 'include_ancestors': True, 'include_inactive': False, 'repair_alternation': False,
            'include_row_ids': False, 'include_compacted': True})['messages']
    assert 'GONE' not in owner
    assert [m['content'] for m in worker] == owner
