"""``messages.transcript_handoff_rows`` on SQLite looks its rows up in a fixed number of indexed reads, so its
cost grows with the rows asked about, not with their product with the rows stored: a whole-lineage page asks
about every row before its window. It finds what the one join it replaced found."""
import json

from agent.context_compressor import SUMMARY_PREFIX
from persistence_assembly import open_persistence, profile_scope
from persistence.sqlite.conversation._store.messages import _row_to_message_dict
from persistence.sqlite.conversation._store.stored_reads import transcript_handoff_rows
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session

# The join the lookup replaced: each shown row, and every earlier row of the lineage with its message_uid,
# that can hold a handoff.
_ONE_JOIN = (
    "SELECT w.id AS _shown_id, g.* FROM messages w JOIN messages g"
    " ON g.session_id IN (SELECT value FROM json_each(:segments)) AND (g.id = w.id OR ("
    "w.message_uid IS NOT NULL AND w.message_uid <> '' AND g.message_uid = w.message_uid AND g.id < w.id))"
    " WHERE w.id IN (SELECT value FROM json_each(:shown)) AND g.role IN ('user', 'assistant')"
    " AND (g._compressed_summary = 1 OR instr(g.content, '[CONTEXT COMPACTION') > 0"
    " OR instr(g.content, '[CONTEXT SUMMARY]') > 0) ORDER BY w.id, g.id")


def _open(home):
    return open_persistence(profile_scope(home=home))


def _append(store, session_id, role, content):
    [row_id] = seed_messages(store, session_id, {'role': role, 'content': content})
    return row_id


def _steps(store, fn):
    """``fn()`` and the thousands of SQLite VM steps it took on the snapshot's one connection."""
    steps = [0]
    with store.messages.read_snapshot():
        conn = raw_holder(store)._pinned_read_conn()
        conn.set_progress_handler(lambda: steps.__setitem__(0, steps[0] + 1) or 0, 1000)
        try:
            return fn(), steps[0]
        finally:
            conn.set_progress_handler(None, 0)


def test_the_handoff_lookup_for_every_row_of_a_long_segment_stays_linear(tmp_path):
    store = _open(tmp_path)
    holder = raw_holder(store)
    try:
        seed_session(store, 's', 'cli')
        for i in range(1500):
            seed_messages(store, 's', {'role': 'user', 'content': f'question {i}'},
                          {'role': 'assistant', 'content': f'answer {i}'})
        carrier = _append(store, 's', 'user', SUMMARY_PREFIX + ' what came before')
        ids = [row['id'] for row in holder._read_all("SELECT id FROM messages WHERE session_id = 's' ORDER BY id")]
        found, steps = _steps(store, lambda: transcript_handoff_rows(holder, ids, ['s']))
        assert list(found) == [carrier] and [message['id'] for message in found[carrier]] == [carrier]
        # About 0.4 thousand steps here; a lookup per shown row over the segment takes about 50 thousand.
        assert steps < 3000, steps
    finally:
        store.close()


def test_the_handoff_lookup_finds_what_one_join_over_the_shown_rows_found(tmp_path):
    """Copies of a carrier in every state: live in an earlier segment, archived and removed in the tip, and
    a later copy, which an earlier shown row never holds."""
    store = _open(tmp_path)
    holder = raw_holder(store)
    try:
        seed_session(store, 'root', 'cli')
        seed_session(store, 'tip', 'cli', parent_session_id='root')
        rows = [('root', 'user', 'question', None, 1, 0), ('root', 'user', SUMMARY_PREFIX + ' one', 'U', 1, 0),
                ('tip', 'user', SUMMARY_PREFIX + ' one, re-wrapped', 'U', 1, 0), ('tip', 'user', 'one, unwrapped', 'U', 1, 0),
                ('tip', 'user', SUMMARY_PREFIX + ' two', 'V', 0, 1), ('tip', 'user', SUMMARY_PREFIX + ' two, removed', 'V', 0, 0),
                ('tip', 'user', 'two, carried', 'V', 1, 0), ('tip', 'assistant', 'answer', None, 1, 0)]
        ids = []
        for session_id, role, content, uid, active, compacted in rows:
            row_id = _append(store, session_id, role, content)
            holder._execute_write(lambda conn, row_id=row_id, uid=uid, active=active, compacted=compacted: conn.execute(
                'UPDATE messages SET message_uid = ?, active = ?, compacted = ? WHERE id = ?', (uid, active, compacted, row_id)))
            ids.append(row_id)
        for shown in (ids, ids[1:4], [ids[1]], ids[3:], [ids[6]]):
            with holder._read_ctx() as conn:
                joined = conn.execute(_ONE_JOIN, {'segments': json.dumps(['root', 'tip']), 'shown': json.dumps(sorted(shown))}).fetchall()
            expected = {}
            for row in joined:
                message = _row_to_message_dict(holder, row, warn_context='session.transcript', summary_flag=True)
                expected.setdefault(int(message.pop('_shown_id')), []).append(message)
            assert transcript_handoff_rows(holder, shown, ['root', 'tip']) == expected, shown
        assert [m['id'] for m in transcript_handoff_rows(holder, [ids[3]], ['root', 'tip'])[ids[3]]] == [ids[1], ids[2]]
    finally:
        store.close()


def test_an_ordinary_shown_row_is_read_for_its_id_uid_and_flag_only(tmp_path):
    """Only a shown row that can hold a handoff itself is read whole; every other shown row (a large image or
    tool result included) is read for its id, uid and flag, so its body is not loaded a second time."""
    store = _open(tmp_path)
    holder = raw_holder(store)
    try:
        seed_session(store, 's', 'cli')
        plain = [_append(store, 's', 'user', 'x' * 200_000), _append(store, 's', 'assistant', 'answer')]
        carrier = _append(store, 's', 'user', SUMMARY_PREFIX + ' what came before')
        statements = []
        with store.messages.read_snapshot():
            conn = holder._pinned_read_conn()
            conn.set_trace_callback(statements.append)
            try:
                found = transcript_handoff_rows(holder, [*plain, carrier], ['s'])
            finally:
                conn.set_trace_callback(None)
        assert list(found) == [carrier] and [message['id'] for message in found[carrier]] == [carrier]
        by_id = [sql for sql in statements if 'WHERE g.id IN' in sql]
        # The shown rows' read carries no bodies; the one whole-row read by id names the carrier alone.
        assert [sql for sql in by_id if 'g.*' not in sql] and [sql for sql in by_id if 'g.*' in sql], statements
        assert all(f"json_each('{json.dumps([carrier])}')" in sql for sql in by_id if 'g.*' in sql), by_id
    finally:
        store.close()
