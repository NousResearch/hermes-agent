"""``session.transcript`` review fixes: a page and the cursor state it stores come from one read
transaction (without a pooled reader: reads no write landed between, else ``reset``); the cursor state covers every displayed segment (an earlier segment's deleted rows reset a
poller, as its reactions do); ``limit`` bounds the rows sent, handoff rows included, and a row's handoffs
split across pages are each sent once.
"""
import pytest

from tests.gateway.test_session_stored_reads import _call, _legacy_owner, _native
from tests.gateway.test_session_stored_reads_bounds import _session
from tests.gateway.test_session_transcript_poll import Reader, _shape


def _read_path(monkeypatch, db, mode):
    """``pooled``: the WAL read pool (one pinned read transaction). ``fallback``: no pooled reader (a
    DELETE-journal or read-only store, a reader that failed to open, the pool's ceiling)."""
    if mode == 'fallback':
        monkeypatch.setattr(db, '_checkout_read_conn', lambda: None)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['pooled', 'fallback'])
async def test_rows_appended_and_deleted_during_a_page_read_reset_the_next_poll(tmp_path, monkeypatch, mode):
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
        _read_path(monkeypatch, db, mode)
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
@pytest.mark.parametrize('mode', ['pooled', 'fallback'])
async def test_a_held_row_deleted_while_its_page_is_read_is_not_kept(tmp_path, monkeypatch, mode):
    """The page holds a row a delete then removes before the cursor state is read: the reader never keeps it."""
    from gateway import session_stored_reads
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        try:
            db.append_message(target, 'user', 'held')
            _read_path(monkeypatch, db, mode)
            original, armed = session_stored_reads.read_page, [True]

            def racing(*args, **kwargs):
                result = original(*args, **kwargs)
                if armed[0]:
                    armed[0] = False
                    db.clear_messages(target)
                return result
            monkeypatch.setattr(session_stored_reads, 'read_page', racing)
            reader = Reader(owner, ref.session_id)
            await reader.poll()
            await reader.poll()
            assert reader.rows == [] and _shape(reader.rows) == await reader.fresh()
        finally:
            await owner.close()


@pytest.mark.asyncio
async def test_a_fallback_read_that_never_settles_resets_from_the_first_page_until_one_does(tmp_path, monkeypatch):
    """No pooled reader, and a write lands during the read and again during its retry: no page is proven
    against a cursor state, so the reply is ``reset`` with the first page, and the next poll resets again
    (its stored state matches no transcript) instead of continuing from it."""
    from gateway import session_stored_reads
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        try:
            for n in range(3):
                db.append_message(target, 'user', f'message {n}')
            reader = Reader(owner, ref.session_id, limit=2)
            await reader.poll()
            _read_path(monkeypatch, db, 'fallback')
            original, writes = session_stored_reads.read_page, [2]

            def busy(*args, **kwargs):
                result = original(*args, **kwargs)
                if writes[0]:
                    writes[0] -= 1
                    db.set_message_reaction(target, result[0][0]['row_id'], '\U0001f44d')
                return result
            monkeypatch.setattr(session_stored_reads, 'read_page', busy)
            db.append_message(target, 'user', 'message 3')
            first = (await _call(owner, 'session.transcript', session_id=ref.session_id, limit=2,
                                 cursor=reader.cursor))['result']
            assert first['reset'] is True and [row.get('text') for row in first['rows']] == ['message 0', 'message 1']
            again = (await _call(owner, 'session.transcript', session_id=ref.session_id, limit=2,
                                 cursor=first['next_cursor']))['result']
            assert again['reset'] is True, again
            reader.rows, reader.cursor = again['rows'], again['next_cursor']
            await reader.poll()
            assert _shape(reader.rows) == await reader.fresh()
            assert [row.get('text') for row in reader.rows] == [f'message {n}' for n in range(4)]
        finally:
            await owner.close()


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
    db.append_message(target, 'user', 'question 1')
    db.append_message(target, 'assistant', 'answer 1')
    db.append_message(target, 'user', 'question 2', message_uid='q2')
    db.append_message(target, 'assistant', 'answer 2')
    for goal in ('first', 'second'):
        _rewrap(db, target, goal)


def _rewrap(db, target, goal):
    """One more in-place compaction that re-wraps question 2's carrier: one more handoff before it."""
    from agent.context_compressor import (
        SUMMARY_PREFIX, _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER, _SUMMARY_END_MARKER,
    )
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


@pytest.mark.asyncio
async def test_a_rewrap_between_newest_first_pages_sends_no_handoff_twice(tmp_path, monkeypatch):
    """A newest-first page ended inside question 2's handoffs (it sent the row and the newer one), then a
    compaction re-wrapped the carrier once more: the next pages send the older handoff, not the newer
    one again, and nothing the walk started with is skipped."""
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        _two_handoffs_on_one_carrier(db, target)
        started = (await _call(owner, 'session.transcript', session_id=ref.session_id, limit=100))['result']['rows']
        reply = (await _call(owner, 'session.transcript', session_id=ref.session_id, limit=3, order='latest'))['result']
        assert [row.get('text') for row in reply['rows']] == ['## Goal\nsecond', 'question 2', 'answer 2'], reply
        _rewrap(db, target, 'third')
        rows = reply['rows']
        while reply['next_cursor']:
            reply = (await _call(owner, 'session.transcript', session_id=ref.session_id, limit=3, order='latest',
                                 cursor=reply['next_cursor']))['result']
            rows = reply['rows'] + rows
        walked = [(row['row_kind'], row.get('text')) for row in rows]
        assert len(walked) == len(set(walked)), walked
        assert {(row['row_kind'], row.get('text')) for row in started} <= set(walked), walked
        await owner.close()
