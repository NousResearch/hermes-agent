"""Display reads hide a message whose newest copy was removed (rewound, or replaced by a newer copy), so
they need the removed rows' identities, but never their bodies: ``assert_resume_safe`` does not count
those rows, and repeated in-place compaction leaves one removed copy of the carried tail per round
(review of #140971). Real SessionDB, real ``archive_and_compact``."""
import sqlite3

from hermes_state import SessionDB

BIG = 'x' * 100_000


def _compacted_repeatedly(db, rounds):
    db.create_session('s', source='test')
    db.append_messages_batch('s', [{'role': 'user', 'content': 'U1'}, {'role': 'assistant', 'content': 'A1'},
                                   {'role': 'user', 'content': 'U2'}, {'role': 'assistant', 'content': BIG}])
    for n in range(rounds):
        # /compress with nothing new since: the same verbatim tail (U2 + the big reply) is carried again.
        db.archive_and_compact('s', [{'role': 'user', 'content': f'[CONTEXT SUMMARY] round {n}'},
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
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        _compacted_repeatedly(db, 6)
        removed = db._read_one("SELECT COUNT(*) FROM messages WHERE content = ? AND active = 0 AND compacted = 0",
                               (BIG,))[0]
        assert removed >= 5  # the fixture really leaves removed copies behind
        archived = db._read_one("SELECT COUNT(*) FROM messages WHERE content = ? AND compacted = 1", (BIG,))[0]
        for read in (
            lambda: db.get_messages_as_conversation('s', include_ancestors=True, include_compacted=True),
            lambda: db.get_resume_conversations('s')[1],
        ):
            shown, bodies = _big_bodies_read(db, monkeypatch, read)
            assert sum(1 for m in shown if m.get('content') == BIG) == 1
            # Only the live copy and the compaction-archived ones (what the resume guard counts) carry a body.
            assert bodies <= 1 + archived, (bodies, archived, removed)


def test_a_rewound_turn_stays_hidden_without_its_body(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        db.append_messages_batch('s', [{'role': 'user', 'content': 'KEEP'}, {'role': 'assistant', 'content': 'OK'},
                                       {'role': 'user', 'content': 'GONE'}, {'role': 'assistant', 'content': BIG}])
        db.rewind_to_message('s', next(m['id'] for m in db.get_messages('s') if m['content'] == 'GONE'))
        shown = db.get_messages_as_conversation('s', include_ancestors=True, include_compacted=True)
        assert [m['content'] for m in shown] == ['KEEP', 'OK']


def test_the_worker_history_read_agrees_with_the_owner_display_read(tmp_path):
    """``worker_history`` (a worker's display read) applies the same removed-copy rule as the owner's."""
    from hermes_state_worker_compression import worker_history
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        db.append_messages_batch('s', [{'role': 'user', 'content': 'KEEP'}, {'role': 'assistant', 'content': 'OK'},
                                       {'role': 'user', 'content': 'GONE'}, {'role': 'assistant', 'content': 'NO'}])
        # Rotated: the parent keeps its copies active, the child's copies of GONE/NO are rewound.
        assert db.try_acquire_compression_lock('s', 'fixture')
        try:
            db.publish_compression_child(parent_session_id='s', child_session_id='c', source='test', messages=[
                {'role': 'user', 'content': 'SUMMARY'}, {'role': 'user', 'content': 'GONE'},
                {'role': 'assistant', 'content': 'NO'}], compression_lock_holder='fixture')
        finally:
            db.release_compression_lock('s', 'fixture')
        db.rewind_to_message('c', next(m['id'] for m in db.get_messages('c') if m['content'] == 'GONE'))
        owner = [m['content'] for m in db.get_messages_as_conversation('c', include_ancestors=True,
                                                                       include_compacted=True)]
        with db._read_ctx() as conn:
            worker = worker_history(db, conn, 'c', {
                'target': 'c', 'include_ancestors': True, 'include_inactive': False, 'repair_alternation': False,
                'include_row_ids': False, 'include_compacted': True})['messages']
        assert 'GONE' not in owner
        assert [m['content'] for m in worker] == owner
