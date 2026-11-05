"""``session.transcript`` consistency (#150091 review): one read generation per page, tool results paired
with their own call occurrence, cursor ints in SQLite's range, summary-carrier boundaries, and the
resume's conversation normalization on SQL-paged rows."""
import base64
import json

import pytest

from persistence.conversation.messages import ConversationRequest
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import conversation
from tests.fixtures.storage_seed import seed_messages
from tests.gateway.test_session_stored_reads import _append, _call, _compact, _compact_carrying_tail, _flat, _pages
from tests.gateway.test_session_stored_reads_bounds import _session


def _row_id(db, session_id, role, content, **extra):
    """The id of one appended message row."""
    [row_id] = seed_messages(db, session_id, {'role': role, 'content': content, **extra})
    return row_id


def _write(db, sql, params):
    raw_holder(db)._execute_write(lambda conn: conn.execute(sql, params))


@pytest.mark.asyncio
async def test_a_compaction_between_the_cursor_lookup_and_the_page_read_skips_nothing(tmp_path, monkeypatch):
    from agent.context_compressor import SUMMARY_PREFIX
    from persistence.sqlite.conversation._store import stored_reads
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        for i in (1, 2, 3):
            _append(db, target, 'user', f'question {i}')
            _append(db, target, 'assistant', f'answer {i}')
        first = await _call(owner, 'session.transcript', session_id=ref.session_id, limit=1, order='latest')
        assert [row['text'] for row in first['result']['rows']] == ['answer 3']
        original, fired = stored_reads._display_rows_from_conn, []

        def compaction_lands_first(conn, *args, **kwargs):
            if not fired:  # commits after the cursor's slot was read, before the page is
                fired.append(True)
                _compact_carrying_tail(db, target, f'{SUMMARY_PREFIX}\n## Goal\none')
            return original(conn, *args, **kwargs)
        monkeypatch.setattr(stored_reads, '_display_rows_from_conn', compaction_lands_first)
        rest = await _call(owner, 'session.transcript', session_id=ref.session_id, limit=10, order='latest',
                           cursor=first['result']['next_cursor'])
        assert fired
        assert [row['text'] for row in rest['result']['rows'] if row['row_kind'] == 'message'] == [
            'question 1', 'answer 1', 'question 2', 'answer 2', 'question 3']
        await owner.close()


def test_a_result_is_labelled_by_its_own_call_when_the_provider_id_repeats(tmp_path, monkeypatch):
    from gateway.session_controls import display_projection
    from gateway.session_stored_reads import _call_context

    def call(path):
        return [{'id': 'c1', 'type': 'function',
                 'function': {'name': 'write_file', 'arguments': json.dumps({'path': path})}}]
    with _session(tmp_path, monkeypatch) as (db, _authority, _owner, _ref, target):
        _append(db, target, 'user', 'first')
        _append(db, target, 'assistant', '', tool_calls=call('f0'))
        _append(db, target, 'tool', 'done 0', tool_call_id='c1')
        _append(db, target, 'user', 'second')
        _append(db, target, 'assistant', '', tool_calls=call('f1'))
        opening = _row_id(db, target, 'tool', 'done 1', tool_call_id='c1')
        _append(db, target, 'assistant', 'ok')
        _append(db, target, 'assistant', '', tool_calls=call('f2'))
        _append(db, target, 'tool', 'done 2', tool_call_id='c1')
        # A page that opens with the result of the f1 call; a later call on it reuses the provider id.
        page = [row for row in conversation(db, target, ConversationRequest(include_row_ids=True))
                if row['_row_id'] >= opening]
        projected = display_projection(_call_context(db, page, [target]) + page, str(tmp_path))
        assert [row['args'] for row in projected if row['role'] == 'tool'] == [{'path': 'f1'}, {'path': 'f2'}]


@pytest.mark.asyncio
async def test_a_cursor_past_sqlites_integer_range_is_invalid_params(tmp_path, monkeypatch):
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        _append(db, target, 'user', 'question 1')
        cursor = base64.urlsafe_b64encode(json.dumps(['t3', ref.session_id, None, 'oldest', [0, 2 ** 63], target, 0,
                                                       [0, 0, 0]]).encode()).decode()
        reply = await _call(owner, 'session.transcript', session_id=ref.session_id, cursor=cursor)
        assert reply.get('error', {}).get('data', {}).get('reason') == 'invalid_params', reply
        await owner.close()


@pytest.mark.asyncio
async def test_a_reply_a_later_compaction_wraps_in_a_summary_carrier_is_not_shown_twice(tmp_path, monkeypatch):
    from agent.context_compressor import (
        SUMMARY_PREFIX, _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER, _SUMMARY_END_MARKER,
    )
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        for i in (1, 2, 3):
            _append(db, target, 'user', f'question {i}')
            _append(db, target, 'assistant', f'answer {i}')

        def wrap_the_pages_last_reply():
            # The carried tail opens with the reply that ended the first page, now a summary carrier.
            reply, *rest = conversation(db, target)[-3:]
            carrier = {**reply, '_compressed_summary': True, 'content': (
                f"{_MERGED_PRIOR_CONTEXT_HEADER}\nanswer 2\n{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_PREFIX}\n"
                f"## Goal\none\n{_SUMMARY_END_MARKER}")}
            _compact(db, target, [carrier, *rest], 3)
        pages = await _pages(owner, ref.session_id, 4, between=wrap_the_pages_last_reply)
        # The compaction resets the reader; its first page's fourth row is the handoff (it counts toward limit).
        assert pages[0] == ['question 1', 'answer 1', 'question 2']
        assert _flat(pages) == ['question 1', 'answer 1', 'question 2', 'answer 2', 'question 3', 'answer 3'], pages
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('order', ['oldest', 'latest'])
async def test_sql_paged_rows_get_the_resumes_normalization_across_a_page_edge(tmp_path, monkeypatch, order):
    import gateway.session_stored_reads as stored
    # No tool-run slack: the SQL window is exactly ``limit`` rows, so the harness prompt and its reply land
    # in two windows (with the default 64 extra rows this whole fixture is one window).
    monkeypatch.setattr(stored, '_TOOL_RUN', 0)
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        _append(db, target, 'user', 'question 1')
        _append(db, target, 'assistant', 'answer 1')
        # An older build wrote the background curator's turn into the session: a harness prompt and its reply.
        _append(db, target, 'user', 'Review the conversation above and consider saving to memory.')
        _append(db, target, 'assistant', 'Saved two memories.')
        _append(db, target, 'user', 'question 2')
        _append(db, target, 'assistant', '[memory]', tool_calls=[{'id': 'm1', 'type': 'function',
                'function': {'name': 'memory', 'arguments': '{}'}}])
        _append(db, target, 'tool', 'saved', tool_call_id='m1')
        _append(db, target, 'assistant', 'answer 2')
        # Three rows a page: the harness prompt ends the first page, its reply opens the second.
        pages = await _pages(owner, ref.session_id, 3, order=order)
        texts = [text for text in _flat(pages if order == 'oldest' else pages[::-1]) if text is not None]
        assert texts == ['question 1', 'answer 1', 'question 2', 'answer 2'], pages
        await owner.close()


def test_a_result_is_labelled_by_its_occurrence_of_a_provider_id_repeated_in_one_row(tmp_path, monkeypatch):
    """Two folded turns in one assistant row call ``c1`` twice; ``tool_call_uids`` holds one uid per occurrence."""
    from gateway.session_controls import display_projection
    from gateway.session_stored_reads import _call_context
    with _session(tmp_path, monkeypatch) as (db, _authority, _owner, _ref, target):
        _append(db, target, 'user', 'write two files')
        calls = [{'id': 'c1', 'type': 'function', 'function': {'name': 'write_file', 'arguments': json.dumps(
            {'path': path})}} for path in ('f1', 'f2')]
        folded = _row_id(db, target, 'assistant', '', tool_calls=calls)
        first = _row_id(db, target, 'tool', 'done 1', tool_call_id='c1')
        second = _row_id(db, target, 'tool', 'done 2', tool_call_id='c1')
        _write(db, 'UPDATE messages SET tool_call_uids = ? WHERE id = ?', (json.dumps({'c1': ['u1', 'u2']}), folded))
        for row_id, uid in ((first, 'u1'), (second, 'u2')):
            _write(db, 'UPDATE messages SET tool_call_uid = ? WHERE id = ?', (uid, row_id))
        # A size cut put the second result alone on the next page.
        page = [row for row in conversation(db, target, ConversationRequest(include_row_ids=True))
                if row['_row_id'] == second]
        projected = display_projection(_call_context(db, page, [target]) + page, str(tmp_path))
        assert [(row['name'], row.get('args')) for row in projected] == [('write_file', {'path': 'f2'})]


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


@pytest.mark.asyncio
async def test_a_compaction_committing_inside_a_lineage_read_drops_nothing(tmp_path, monkeypatch):
    """A rotated lineage reads its display rows and its removed rows' identities; a compaction that carries
    the tip's last reply commits between those two statements. Both must see one generation."""
    from agent.context_compressor import SUMMARY_PREFIX
    from gateway.session_authority import initialize_session_authority
    from tests.gateway.test_session_stored_reads import _legacy_owner, _native

    store, db, runner = _legacy_owner(tmp_path, monkeypatch)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = _native(authority, tmp_path, 'native')
    holder = raw_holder(db)
    original, fired = holder._read_retrying_ioerr, []

    class Conn:
        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def execute(self, sql, *args):
            cursor = self._conn.execute(sql, *args)
            if fired or 'compacted, display_order' not in sql or 'tui-tip' not in str(args):
                return cursor
            rows = cursor.fetchall()  # the display rows are read; the removed rows' read comes next
            fired.append(True)
            _compact_carrying_tail(db, 'tui-tip', f'{SUMMARY_PREFIX}\n## Goal\nrotated', tail_rows=1)
            return _Rows(rows)
    monkeypatch.setattr(holder, '_read_retrying_ioerr', lambda fn: original(lambda conn: fn(Conn(conn))))
    try:
        assert _flat(await _pages(native, 'tui-tip', 10)) == [
            'before compression', 'old answer', 'after compression', 'new answer']
        assert fired
    finally:
        await native.close()
        store.close_all_db_handles()
