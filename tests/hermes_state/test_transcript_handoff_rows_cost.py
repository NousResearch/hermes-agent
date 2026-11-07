"""``SessionDB.transcript_handoff_rows`` looks its rows up in a fixed number of indexed reads, so its cost
grows with the rows asked about, not with their product with the rows stored: a whole-lineage page asks
about every row before its window."""
from agent.context_compressor import SUMMARY_PREFIX
from hermes_state import SessionDB


def _steps(db, fn):
    """``fn()`` and the thousands of SQLite VM steps it took on the snapshot's one connection."""
    steps = [0]
    with db.transcript_read_snapshot():
        conn = db._pinned_read_conn()
        conn.set_progress_handler(lambda: steps.__setitem__(0, steps[0] + 1) or 0, 1000)
        try:
            return fn(), steps[0]
        finally:
            conn.set_progress_handler(None, 0)


def test_the_handoff_lookup_for_every_row_of_a_long_segment_stays_linear(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='cli')
        for i in range(1500):
            db.append_message('s', 'user', f'question {i}')
            db.append_message('s', 'assistant', f'answer {i}')
        carrier = db.append_message('s', 'user', SUMMARY_PREFIX + ' what came before')
        ids = [row['id'] for row in db._read_all("SELECT id FROM messages WHERE session_id = 's' ORDER BY id")]
        found, steps = _steps(db, lambda: db.transcript_handoff_rows(ids, ['s']))
        assert list(found) == [carrier] and [message['id'] for message in found[carrier]] == [carrier]
        # About 0.4 thousand steps here; a lookup per shown row over the segment takes about 50 thousand.
        assert steps < 3000, steps
    finally:
        db.close()
