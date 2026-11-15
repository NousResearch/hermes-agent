"""A display read hashes a display identity only where a removed copy could match it (RT-39).

``drop_removed`` hides a message whose newest copy was removed. Hashing every live row's display key to
compare against the removed rows cost a warm 30k-row resume ~0.17 s (+12% end to end), though a live row
can only share a removed row's key at one of the removed rows' timestamps. The guard counts hashes, not
seconds. Real SessionDB, real rewind.
"""
from hermes_state import SessionDB

LIVE_TURNS = 200


def _session(db, *, rewind):
    db.create_session('s', source='test')
    msgs = []
    for i in range(LIVE_TURNS):
        msgs += [{'role': 'user', 'content': f'U{i}', 'timestamp': 1_000_000 + 2 * i},
                 {'role': 'assistant', 'content': f'A{i}', 'timestamp': 1_000_001 + 2 * i}]
    db.append_messages_batch('s', msgs)
    if rewind:
        db.append_messages_batch('s', [{'role': 'user', 'content': 'GONE', 'timestamp': 2_000_000},
                                       {'role': 'assistant', 'content': 'NO', 'timestamp': 2_000_001}])
        db.rewind_to_message('s', next(m['id'] for m in db.get_messages('s') if m['content'] == 'GONE'))


def _hashes(db, monkeypatch, read):
    calls = []
    real = SessionDB._display_identity

    def spy(key):
        calls.append(key)
        return real(key)
    monkeypatch.setattr(SessionDB, '_display_identity', staticmethod(spy))
    shown = read()
    monkeypatch.undo()
    return shown, len(calls)


def _reads(db):
    return (lambda: db.get_messages_as_conversation('s', include_ancestors=True, include_compacted=True),
            lambda: db.get_resume_conversations('s')[1],
            lambda: db.get_messages_as_conversation('s', include_compacted=True))


def test_a_display_read_with_no_removed_row_hashes_nothing(tmp_path, monkeypatch):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        _session(db, rewind=False)
        for read in _reads(db):
            shown, hashes = _hashes(db, monkeypatch, read)
            assert len(shown) == 2 * LIVE_TURNS
            assert hashes == 0


def test_a_display_read_hashes_only_rows_at_a_removed_rows_timestamp(tmp_path, monkeypatch):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        _session(db, rewind=True)
        removed = db._read_one('SELECT COUNT(*) FROM messages WHERE active = 0 AND compacted = 0')[0]
        assert removed == 2
        for read in _reads(db):
            shown, hashes = _hashes(db, monkeypatch, read)
            assert [m['content'] for m in shown][-2:] == [f'U{LIVE_TURNS - 1}', f'A{LIVE_TURNS - 1}']
            assert 'GONE' not in [m['content'] for m in shown]
            # At most the removed rows themselves (a legacy row without a stored identity); no live row shares
            # their timestamps here.
            assert hashes <= removed, hashes
