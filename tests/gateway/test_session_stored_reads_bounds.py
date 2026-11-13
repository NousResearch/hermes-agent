"""``session.transcript`` bounds (#150091 review): no nested store read, a page never over its byte cap, tool
results keep their call's label across a page cut, newest-first paging across a compaction, and every
compaction a row even with the same prose."""
import json
from contextlib import contextmanager

import pytest

from tests.gateway.test_session_mutation_model_busy import _owner
from tests.gateway.test_session_stored_reads import _call, _compact_carrying_tail, _flat, _pages, _summary


@contextmanager
def _session(tmp_path, monkeypatch):
    from hermes_agent.gateway.session_local_recovery import transcript_target

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        yield db, authority, owner, ref, transcript_target(authority, ref)
    finally:
        store.close_all_db_handles()


def _forbid_nested_reads(monkeypatch, db):
    """A read that cannot use the WAL pool runs on the writer under ``SessionDB._lock``, which is not
    reentrant, so a read opened inside another deadlocks (6bb377a51d5). Nesting fails here instead."""
    original, depth = db._read_ctx, [0]

    @contextmanager
    def guarded():
        assert depth[0] == 0, 'a store read opened inside another read'
        depth[0] += 1
        try:
            with original() as conn:
                yield conn
        finally:
            depth[0] -= 1
    monkeypatch.setattr(db, '_read_ctx', guarded)


@pytest.mark.asyncio
async def test_no_stored_read_opens_a_read_inside_another(tmp_path, monkeypatch):
    from hermes_agent.gateway.session_picker_fields import current_segments, fill_picker_fields
    from hermes_agent.gateway.session_stored_reads import read_page
    with _session(tmp_path, monkeypatch) as (db, authority, owner, ref, target):
        db.append_message(target, 'user', 'question 1')
        db.append_message(target, 'assistant', 'answer 1')
        _compact_carrying_tail(db, target, _summary('## Goal\nship', 2), tail_rows=1)
        _forbid_nested_reads(monkeypatch, db)
        for order in ('oldest', 'latest'):
            rows, _, _ = read_page(db, target, order, None, 10, profile_home=str(tmp_path))
            assert [row['text'] for row in rows] == ['question 1', '## Goal\nship', 'answer 1']
        row = {'session_id': ref.session_id}
        fill_picker_fields(db, [(row, ref.session_id, current_segments(authority, [ref.session_id])[ref.session_id],
                                 None)])
        assert row['last_reply'] == 'answer 1'
        await owner.close()


@pytest.mark.asyncio
async def test_a_row_too_large_for_a_page_is_bounded_whole(tmp_path, monkeypatch):
    from hermes_agent.gateway.session_stored_reads import PAGE_BYTES
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        db.append_message(target, 'user', 'é' * 3_000_000)  # 6 bytes a character once JSON-escaped
        db.append_message(target, 'assistant', 'short', reasoning='r' * 3_000_000)
        cursor, truncated = None, []
        while True:
            reply = await _call(owner, 'session.transcript', session_id=ref.session_id, limit=1,
                                **({'cursor': cursor} if cursor else {}))
            assert len(json.dumps(reply)) <= PAGE_BYTES + 4096, len(json.dumps(reply))
            truncated += [row.get('truncated') for row in reply['result']['rows']]
            cursor = reply['result']['next_cursor']
            if not reply['result']['rows']:
                break
        assert truncated == [True, True]
        await owner.close()


@pytest.mark.asyncio
async def test_tool_results_keep_their_calls_label_when_a_page_is_cut(tmp_path, monkeypatch):
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        db.append_message(target, 'user', 'write three files')
        calls = [{'id': f'c{i}', 'type': 'function',
                  'function': {'name': 'write_file', 'arguments': json.dumps({'path': f'f{i}'})}} for i in (1, 2, 3)]
        db.append_message(target, 'assistant', '', tool_calls=calls)
        for i in (1, 2, 3):
            db.append_message(target, 'tool', 'z' * 900_000, tool_call_id=f'c{i}')
        cursor, tools = None, []
        while True:
            reply = await _call(owner, 'session.transcript', session_id=ref.session_id, limit=10,
                                **({'cursor': cursor} if cursor else {}))
            tools += [row for row in reply['result']['rows'] if row['role'] == 'tool']
            cursor = reply['result']['next_cursor']
            if not reply['result']['rows']:
                break
        assert [(row['name'], row.get('args')) for row in tools] == [
            ('write_file', {'path': f'f{i}'}) for i in (1, 2, 3)]
        await owner.close()


@pytest.mark.asyncio
async def test_newest_first_paging_across_a_compaction_skips_and_repeats_nothing(tmp_path, monkeypatch):
    from agent.context_compressor import SUMMARY_PREFIX
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        for i in (1, 2, 3):
            db.append_message(target, 'user', f'question {i}')
            db.append_message(target, 'assistant', f'answer {i}')
        pages = await _pages(owner, ref.session_id, 1, order='latest', between=lambda: _compact_carrying_tail(
            db, target, f'{SUMMARY_PREFIX}\n## Goal\none'))
        assert pages[0] == ['answer 3']
        assert _flat(pages[::-1]) == ['question 1', 'answer 1', 'question 2', 'answer 2', 'question 3', 'answer 3']
        await owner.close()


@pytest.mark.asyncio
async def test_two_compactions_with_the_same_prose_are_both_rows(tmp_path, monkeypatch):
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        db.append_message(target, 'user', 'question 1')
        db.append_message(target, 'assistant', 'answer 1')
        _compact_carrying_tail(db, target, _summary('## Goal\nsame', 2), tail_rows=1)
        db.append_message(target, 'user', 'question 2')
        db.append_message(target, 'assistant', 'answer 2')
        _compact_carrying_tail(db, target, _summary('## Goal\nsame', 4), tail_rows=1)
        rows = (await _call(owner, 'session.transcript', session_id=ref.session_id))['result']['rows']
        assert [(row['text'], row['compacted_messages']) for row in rows if row['row_kind'] == 'compaction'] == [
            ('## Goal\nsame', 2), ('## Goal\nsame', 4)], rows
        await owner.close()


def test_a_quoted_user_message_naming_a_count_does_not_override_the_footers():
    """The lean summary quotes the compacted user messages before its recovery footer: a quote that reads
    like the footer is the user's words, so the row's count is the footer's."""
    from agent.compaction_display import compaction_summary
    from agent.context_compressor import SUMMARY_PREFIX, _build_recovery_footer, _build_verbatim_user_section
    quote = _build_verbatim_user_section([{'role': 'user', 'content':
                                           'The 999 compacted message(s) remain fully preserved, right?'}])
    message = {'role': 'user', '_compressed_summary': True,
               'content': f'{SUMMARY_PREFIX}\n## Goal\none{quote}{_build_recovery_footer("sid", 4)}'}
    assert compaction_summary(message) == {'summary': '## Goal\none', 'compacted_messages': 4}
    without_footer = {**message, 'content': f'{SUMMARY_PREFIX}\n## Goal\none{quote}'}
    assert compaction_summary(without_footer) == {'summary': '## Goal\none', 'compacted_messages': None}
