"""``session.transcript`` review fixes: a page and the cursor state it stores come from one read
transaction; the cursor state covers every displayed segment (an earlier segment's deleted rows reset a
poller, as its reactions do); ``limit`` bounds the rows sent, handoff rows included, and a row's handoffs
split across pages are each sent once.
"""
import pytest

from tests.gateway.test_session_stored_reads import _call, _legacy_owner, _native
from tests.gateway.test_session_stored_reads_bounds import _session
from tests.gateway.test_session_transcript_poll import Reader, _shape


@pytest.mark.asyncio
async def test_rows_appended_and_deleted_during_a_page_read_reset_the_next_poll(tmp_path, monkeypatch):
    """A row appended after the poll started is on the page, then deleted before the state is stored: the
    cursor must vouch for the page's own generation, so the next poll resets instead of keeping it forever."""
    from gateway import session_stored_reads
    from gateway.session_local_recovery import transcript_target
    from tests.gateway.test_session_mutation_model_busy import _owner

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        target = transcript_target(authority, ref)
        reader = Reader(owner, ref.session_id)
        await reader.poll()
        original, armed = session_stored_reads.read_page, [True]

        def racing(*args, **kwargs):
            if not armed[0]:
                return original(*args, **kwargs)
            armed[0] = False
            db.append_message(target, 'user', 'transient')
            result = original(*args, **kwargs)
            db.clear_messages(target)
            return result
        monkeypatch.setattr(session_stored_reads, 'read_page', racing)
        await reader.poll()
        assert _shape(reader.rows) == await reader.fresh() == []
        db.append_message(target, 'user', 'after')
        await reader.poll()
        assert [row.get('text') for row in reader.rows] == ['after']
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_deleting_an_earlier_segments_rows_resets_the_poller(tmp_path, monkeypatch):
    from gateway.session_authority import initialize_session_authority
    store, db, runner = _legacy_owner(tmp_path, monkeypatch)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = _native(authority, tmp_path, 'native')
    try:
        reader = Reader(native, 'old-tui')
        await reader.poll()
        assert 'before compression' in [row.get('text') for row in reader.rows]
        db.clear_messages('old-tui')
        reader.resets.clear()
        await reader.poll()
        assert reader.resets[0] is True, reader.resets
        assert 'before compression' not in [row.get('text') for row in reader.rows]
        assert _shape(reader.rows) == await reader.fresh()
    finally:
        await native.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_reaction_on_an_earlier_segments_row_resets_the_poller(tmp_path, monkeypatch):
    from gateway.session_authority import initialize_session_authority
    store, db, runner = _legacy_owner(tmp_path, monkeypatch)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = _native(authority, tmp_path, 'native')
    try:
        reader = Reader(native, 'old-tui')
        await reader.poll()
        held = next(row['row_id'] for row in reader.rows if row.get('text') == 'old answer')
        db.set_message_reaction('old-tui', held, '\U0001f44d')
        reader.resets.clear()
        await reader.poll()
        assert reader.resets[0] is True, reader.resets
        assert _shape(reader.rows) == await reader.fresh()
    finally:
        await native.close()
        store.close_all_db_handles()


def _two_handoffs_on_one_carrier(db, target):
    """question 1, answer 1, then question 2 wrapped by two compactions in turn (the second re-wraps the
    first's carrier), then answer 2: the carrier shows two handoffs before it."""
    from agent.context_compressor import (
        SUMMARY_PREFIX, _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER, _SUMMARY_END_MARKER,
    )
    db.append_message(target, 'user', 'question 1')
    db.append_message(target, 'assistant', 'answer 1')
    db.append_message(target, 'user', 'question 2', message_uid='q2')
    db.append_message(target, 'assistant', 'answer 2')
    for goal in ('first', 'second'):
        held = db.get_messages_as_conversation(target)
        carrier = next(row for row in held if row.get('message_uid') == 'q2')
        carrier = {**carrier, '_compressed_summary': True, 'content': (
            f'{_MERGED_PRIOR_CONTEXT_HEADER}\nquestion 2\n{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_PREFIX}\n'
            f'## Goal\n{goal}\n{_SUMMARY_END_MARKER}')}
        db.archive_and_compact(target, [carrier, held[-1]], watermark=db.get_active_message_watermark(target),
                               tail_count=2)


async def _every_page(owner, session_id, limit, order):
    rows, cursor = [], None
    for _ in range(50):
        reply = (await _call(owner, 'session.transcript', session_id=session_id, limit=limit, order=order,
                             **({'cursor': cursor} if cursor else {})))['result']
        assert len(reply['rows']) <= limit, reply['rows']
        rows = reply['rows'] + rows if order == 'latest' else rows + reply['rows']
        cursor = reply['next_cursor']
        if (order == 'latest' and cursor is None) or (order == 'oldest' and not reply['rows']):
            return rows
    raise AssertionError('the pages never ended')


@pytest.mark.asyncio
@pytest.mark.parametrize('order', ['oldest', 'latest'])
@pytest.mark.parametrize('limit', [1, 2])
async def test_limit_counts_handoff_rows_and_a_split_carrier_sends_each_handoff_once(tmp_path, monkeypatch,
                                                                                     order, limit):
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        _two_handoffs_on_one_carrier(db, target)
        whole = (await _call(owner, 'session.transcript', session_id=ref.session_id, limit=100))['result']['rows']
        assert [(row['row_kind'], row.get('text')) for row in whole] == [
            ('message', 'question 1'), ('message', 'answer 1'), ('compaction', '## Goal\nfirst'),
            ('compaction', '## Goal\nsecond'), ('message', 'question 2'), ('message', 'answer 2')], whole
        assert await _every_page(owner, ref.session_id, limit, order) == whole
        await owner.close()
