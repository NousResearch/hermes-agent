"""``session.transcript`` consistency (#150091 review): one read generation per page, tool results paired
with their own call occurrence, cursor ints in SQLite's range, summary-carrier boundaries, and the
resume's conversation normalization on SQL-paged rows."""
import base64
import json

import pytest

from tests.gateway.test_session_stored_reads import _call, _compact_carrying_tail, _flat, _pages
from tests.gateway.test_session_stored_reads_bounds import _session


@pytest.mark.asyncio
async def test_a_compaction_between_the_cursor_lookup_and_the_page_read_skips_nothing(tmp_path, monkeypatch):
    from agent.context_compressor import SUMMARY_PREFIX
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        for i in (1, 2, 3):
            db.append_message(target, 'user', f'question {i}')
            db.append_message(target, 'assistant', f'answer {i}')
        first = await _call(owner, 'session.transcript', session_id=ref.session_id, limit=1, order='latest')
        assert [row['text'] for row in first['result']['rows']] == ['answer 3']
        original, fired = db._display_rows_from_conn, []

        def compaction_lands_first(conn, *args, **kwargs):
            if not fired:  # commits after the cursor's slot was read, before the page is
                fired.append(True)
                _compact_carrying_tail(db, target, f'{SUMMARY_PREFIX}\n## Goal\none')
            return original(conn, *args, **kwargs)
        monkeypatch.setattr(db, '_display_rows_from_conn', compaction_lands_first)
        rest = await _call(owner, 'session.transcript', session_id=ref.session_id, limit=10, order='latest',
                           cursor=first['result']['next_cursor'])
        assert fired
        assert [row['text'] for row in rest['result']['rows']] == [
            'question 1', 'answer 1', 'question 2', 'answer 2', 'question 3']
        await owner.close()


def test_a_result_is_labelled_by_its_own_call_when_the_provider_id_repeats(tmp_path, monkeypatch):
    from gateway.session_controls import display_projection
    from gateway.session_stored_reads import _call_context

    def call(path):
        return [{'id': 'c1', 'type': 'function',
                 'function': {'name': 'write_file', 'arguments': json.dumps({'path': path})}}]
    with _session(tmp_path, monkeypatch) as (db, _authority, _owner, _ref, target):
        db.append_message(target, 'user', 'first')
        db.append_message(target, 'assistant', '', tool_calls=call('f0'))
        db.append_message(target, 'tool', 'done 0', tool_call_id='c1')
        db.append_message(target, 'user', 'second')
        db.append_message(target, 'assistant', '', tool_calls=call('f1'))
        opening = db.append_message(target, 'tool', 'done 1', tool_call_id='c1')
        db.append_message(target, 'assistant', 'ok')
        db.append_message(target, 'assistant', '', tool_calls=call('f2'))
        db.append_message(target, 'tool', 'done 2', tool_call_id='c1')
        # A page that opens with the result of the f1 call; a later call on it reuses the provider id.
        page = [row for row in db.get_messages_as_conversation(target, include_row_ids=True)
                if row['_row_id'] >= opening]
        projected = display_projection(_call_context(db, page, [target]) + page, str(tmp_path))
        assert [row['args'] for row in projected if row['role'] == 'tool'] == [{'path': 'f1'}, {'path': 'f2'}]


@pytest.mark.asyncio
async def test_a_cursor_past_sqlites_integer_range_is_invalid_params(tmp_path, monkeypatch):
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        db.append_message(target, 'user', 'question 1')
        cursor = base64.urlsafe_b64encode(json.dumps(['t2', ref.session_id, 'oldest', [0, 2 ** 63]]).encode()).decode()
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
            db.append_message(target, 'user', f'question {i}')
            db.append_message(target, 'assistant', f'answer {i}')

        def wrap_the_pages_last_reply():
            # The carried tail opens with the reply that ended the first page, now a summary carrier.
            reply, *rest = db.get_messages_as_conversation(target)[-3:]
            carrier = {**reply, '_compressed_summary': True, 'content': (
                f"{_MERGED_PRIOR_CONTEXT_HEADER}\nanswer 2\n{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_PREFIX}\n"
                f"## Goal\none\n{_SUMMARY_END_MARKER}")}
            db.archive_and_compact(target, [carrier, *rest], watermark=db.get_active_message_watermark(target),
                                   tail_count=3)
        pages = await _pages(owner, ref.session_id, 4, between=wrap_the_pages_last_reply)
        assert pages[0] == ['question 1', 'answer 1', 'question 2', 'answer 2']
        assert _flat(pages) == ['question 1', 'answer 1', 'question 2', 'answer 2', 'question 3', 'answer 3'], pages
        await owner.close()


@pytest.mark.asyncio
async def test_sql_paged_rows_get_the_resumes_normalization_across_a_page_edge(tmp_path, monkeypatch):
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        db.append_message(target, 'user', 'question 1')
        db.append_message(target, 'assistant', 'answer 1')
        # An older build wrote the background curator's turn into the session: a harness prompt and its reply.
        db.append_message(target, 'user', 'Review the conversation above and consider saving to memory.')
        db.append_message(target, 'assistant', 'Saved two memories.')
        db.append_message(target, 'user', 'question 2')
        db.append_message(target, 'assistant', '[memory]', tool_calls=[{'id': 'm1', 'type': 'function',
                          'function': {'name': 'memory', 'arguments': '{}'}}])
        db.append_message(target, 'tool', 'saved', tool_call_id='m1')
        db.append_message(target, 'assistant', 'answer 2')
        # Three rows a page: the harness prompt ends the first page, its reply opens the second.
        texts = [text for text in _flat(await _pages(owner, ref.session_id, 3)) if text is not None]
        assert texts == ['question 1', 'answer 1', 'question 2', 'answer 2'], texts
        await owner.close()


def test_a_result_is_labelled_by_its_occurrence_of_a_provider_id_repeated_in_one_row(tmp_path, monkeypatch):
    """Two folded turns in one assistant row call ``c1`` twice; ``tool_call_uids`` holds one uid per occurrence."""
    from gateway.session_controls import display_projection
    from gateway.session_stored_reads import _call_context
    with _session(tmp_path, monkeypatch) as (db, _authority, _owner, _ref, target):
        db.append_message(target, 'user', 'write two files')
        calls = [{'id': 'c1', 'type': 'function', 'function': {'name': 'write_file', 'arguments': json.dumps(
            {'path': path})}} for path in ('f1', 'f2')]
        folded = db.append_message(target, 'assistant', '', tool_calls=calls)
        first = db.append_message(target, 'tool', 'done 1', tool_call_id='c1')
        second = db.append_message(target, 'tool', 'done 2', tool_call_id='c1')
        db._write_sql('UPDATE messages SET tool_call_uids = ? WHERE id = ?', (json.dumps({'c1': ['u1', 'u2']}), folded))
        for row_id, uid in ((first, 'u1'), (second, 'u2')):
            db._write_sql('UPDATE messages SET tool_call_uid = ? WHERE id = ?', (uid, row_id))
        # A size cut put the second result alone on the next page.
        page = [row for row in db.get_messages_as_conversation(target, include_row_ids=True) if row['_row_id'] == second]
        projected = display_projection(_call_context(db, page, [target]) + page, str(tmp_path))
        assert [(row['name'], row.get('args')) for row in projected] == [('write_file', {'path': 'f2'})]
