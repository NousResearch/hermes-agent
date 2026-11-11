"""``session.transcript`` shows each compaction handoff once, before the first row in the whole transcript
that holds it, wherever a page starts: paged at any ``limit``, either order, the rows are exactly a full
read's, across identical uid-less handoffs on several carriers, several handoffs on one carrier, in-place
compactions and rotations. A re-wrap landing between two pages of a walk repeats no row in either order.
"""
import json
import random
from collections import Counter

import pytest

from tests.gateway.test_session_stored_reads import _call, _compact_carrying_tail, _summary
from tests.gateway.test_session_stored_reads_bounds import _session
from tests.gateway.test_session_transcript_snapshot import _every_page, _two_handoffs_on_one_carrier


def _merged(text, goal):
    from agent.context_compressor import (
        SUMMARY_PREFIX, _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER, _SUMMARY_END_MARKER,
    )
    return (f'{_MERGED_PRIOR_CONTEXT_HEADER}\n{text}\n{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_PREFIX}\n'
            f'## Goal\n{goal}\n{_SUMMARY_END_MARKER}')


def _drop_uid_of_newest(db, segment):
    db._write_sql('UPDATE messages SET message_uid = NULL WHERE id = (SELECT MAX(id) FROM messages WHERE session_id = ?)',
                  (segment,))


def _seeded_transcript(db, segment, seed):
    """Random turns, carriers (some uid-less, goals drawn from two, so identical handoffs repeat), in-place
    compactions, a re-wrap that puts a second handoff on a carrier, and rotations into a new segment."""
    rng = random.Random(seed)
    rotations = 0
    for step in range(rng.randint(6, 10)):
        op = rng.choice(['say', 'carrier', 'carrier', 'compact', 'rewrap', 'rotate'])
        live = db.get_messages_as_conversation(segment)
        if op == 'say' or len(live) < 3:
            db.append_message(segment, 'user', f'question {step}', message_uid=f'q{seed}-{step}')
            db.append_message(segment, 'assistant', f'answer {step}')
        elif op == 'carrier':
            db.append_message(segment, rng.choice(['user', 'assistant']), _merged('wrapped', rng.choice('ab')),
                              _compressed_summary=True)
            if rng.random() < 0.6:
                _drop_uid_of_newest(db, segment)
        elif op == 'compact':
            _compact_carrying_tail(db, segment, _summary(f'## Goal\n{rng.choice("ab")}', len(live)), tail_rows=2)
        elif op == 'rewrap' and live[-2].get('message_uid'):
            carrier = {**live[-2], '_compressed_summary': True, 'content': _merged(live[-2].get('content'), rng.choice('ab'))}
            db.archive_and_compact(segment, [carrier, live[-1]], watermark=db.get_active_message_watermark(segment),
                                   tail_count=2)
        elif op == 'rotate':
            child = f'{segment}-r{step}'
            assert db.try_acquire_compression_lock(segment, 'fixture')
            try:
                db.publish_compression_child(
                    parent_session_id=segment, child_session_id=child, source=db.get_session(segment)['source'],
                    model='m', compression_lock_holder='fixture', messages=[
                        {'role': 'user', 'content': _merged('carried', rng.choice('ab')), '_compressed_summary': True},
                        {'role': 'assistant', 'content': f'rotated {step}'}])
            finally:
                db.release_compression_lock(segment, 'fixture')
            segment, rotations = child, rotations + 1
    return rotations, segment


def _inner(content):
    """The wrapped text of a carrier's content (a plain row's own text)."""
    from agent.context_compressor import _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER
    while isinstance(content, str) and content.startswith(_MERGED_PRIOR_CONTEXT_HEADER):
        content = content[len(_MERGED_PRIOR_CONTEXT_HEADER) + 1:].split(f'\n{_MERGED_SUMMARY_DELIMITER}', 1)[0]
    return content


def _rewrap_newest(db, segment, rng):
    """An in-place compaction that re-wraps the newest carrier-able row (the one before the last): one more
    handoff on it, added next to the row (stored id order), the way a re-wrap lands between two pages."""
    live = db.get_messages_as_conversation(segment)
    if len(live) < 2 or not live[-2].get('message_uid'):
        db.append_message(segment, 'user', 'question late', message_uid=f'late-{rng.random()}')
        db.append_message(segment, 'assistant', 'answer late')
        live = db.get_messages_as_conversation(segment)
    carrier = {**live[-2], '_compressed_summary': True,
               'content': _merged(_inner(live[-2].get('content')), rng.choice(['a', 'b', 'c']))}
    db.archive_and_compact(segment, [carrier, live[-1]], watermark=db.get_active_message_watermark(segment),
                           tail_count=2)


async def _walk_across(owner, session_id, limit, order, at, change):
    """One walk of every page with *change()* run before page *at*. Oldest order is the documented poller (a
    reset drops what it holds; it reads until an empty page after the change); newest first prepends."""
    rows, cursor = [], None
    for page in range(80):
        if page == at:
            change()
        reply = (await _call(owner, 'session.transcript', session_id=session_id, limit=limit, order=order,
                             **({'cursor': cursor} if cursor else {})))['result']
        cursor = reply['next_cursor']
        if order == 'latest':
            rows = reply['rows'] + rows
            if cursor is None:
                return rows
        else:
            rows = (reply['rows'] if reply['reset'] else rows + reply['rows'])
            if not reply['rows'] and page >= at:
                return rows
    raise AssertionError('the pages never ended')


async def _assert_a_rewrap_mid_walk_repeats_nothing(db, owner, session_id, tip, limit, order, rng, at=None):
    """Oldest first, the walk ends holding exactly a fresh read (the re-wrap resets it). Newest first (no reset)
    it repeats no row and skips none it started with; the new handoff is a fresh read's, as an append is."""
    def full():
        return _call(owner, 'session.transcript', session_id=session_id, limit=500)
    started = (await full())['result']['rows']
    if at is None:
        at = rng.randint(1, max(1, -(-len(started) // limit) - 1))
    fired = []
    walked = await _walk_across(owner, session_id, limit, order, at, lambda: fired.append(_rewrap_newest(db, tip, rng)))
    assert fired, (order, limit, at)  # the walk reached page *at*, so the re-wrap really landed mid-walk
    after = (await full())['result']['rows']
    if order == 'oldest':
        assert walked == after, (limit, at)
        return
    def ident(row):
        return json.dumps(row, sort_keys=True, default=str)
    allowed = Counter(map(ident, started)) | Counter(map(ident, after))
    assert not Counter(map(ident, walked)) - allowed, (limit, at, walked)
    def text(rows):
        return Counter((row['row_kind'], row.get('text')) for row in rows)
    # A carried message's old and new copies differ in row_id, so the identity check above lets both through.
    assert not text(walked) - (text(started) | text(after)), (limit, at, walked)
    assert not text(started) - text(walked), (limit, at, walked)


@pytest.mark.asyncio
@pytest.mark.parametrize('seed', range(8))
async def test_every_limit_and_order_pages_exactly_the_full_read(tmp_path, monkeypatch, seed):
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        try:
            _, tip = _seeded_transcript(db, target, seed)
            whole = (await _call(owner, 'session.transcript', session_id=ref.session_id, limit=500))['result']['rows']
            assert any(row['row_kind'] == 'compaction' for row in whole), whole
            for order in ('oldest', 'latest'):
                for limit in range(1, 6):
                    assert await _every_page(owner, ref.session_id, limit, order) == whole, (order, limit)
            rng = random.Random(seed)
            for order in ('oldest', 'latest'):
                for limit in range(1, 6):
                    await _assert_a_rewrap_mid_walk_repeats_nothing(db, owner, ref.session_id, tip, limit, order, rng)
        finally:
            await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('order', ['oldest', 'latest'])
@pytest.mark.parametrize('limit', [1, 2])
async def test_identical_uidless_handoffs_on_two_carriers_are_one_row(tmp_path, monkeypatch, order, limit):
    """Two carriers hold the same uid-less handoff: it is shown once, before the first, on every page split."""
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        try:
            for role in ('user', 'assistant'):
                db.append_message(target, role, _merged('wrapped text', 'same'), _compressed_summary=True)
            db._write_sql('UPDATE messages SET message_uid = NULL WHERE session_id = ?', (target,))
            whole = (await _call(owner, 'session.transcript', session_id=ref.session_id, limit=100))['result']['rows']
            assert [row['row_kind'] for row in whole] == ['compaction', 'message', 'message'], whole
            assert await _every_page(owner, ref.session_id, limit, order) == whole
        finally:
            await owner.close()


@pytest.mark.asyncio
# Six items: a newest-first walk ends on its last page, so the re-wrap lands before one of the others; an
# oldest-first walk reads one more, empty page.
@pytest.mark.parametrize(('order', 'limit', 'at'), [(order, limit, at) for order in ('oldest', 'latest')
                                                    for limit in range(1, 6)
                                                    for at in range(1, -(-6 // limit) + (order == 'oldest'))])
async def test_a_rewrap_before_any_page_of_a_split_carrier_walk_repeats_nothing(tmp_path, monkeypatch, order,
                                                                                limit, at):
    """Question 2's carrier holds two handoffs; a third re-wrap lands before page *at* of the walk."""
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        try:
            _two_handoffs_on_one_carrier(db, target)
            await _assert_a_rewrap_mid_walk_repeats_nothing(
                db, owner, ref.session_id, target, limit, order, random.Random(at), at)
        finally:
            await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('order', ['oldest', 'latest'])
@pytest.mark.parametrize('limit', [1, 2])
@pytest.mark.parametrize('blocks', [False, True])
async def test_two_uidless_carriers_of_one_handoff_show_it_once_whatever_their_content_shape(tmp_path, monkeypatch,
                                                                                             blocks, limit, order):
    """Two uid-less carriers hold the same handoff, as plain text or as content blocks (a compaction merged
    into a structured tail keeps the blocks). Paged in either order, the handoff shows once, as a full read
    shows it."""
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        try:
            text = _merged('wrapped text', 'same')
            content = [{'type': 'text', 'text': text}] if blocks else text
            for role in ('user', 'assistant'):
                db.append_message(target, role, content, _compressed_summary=True)
            db._write_sql('UPDATE messages SET message_uid = NULL WHERE session_id = ?', (target,))
            whole = (await _call(owner, 'session.transcript', session_id=ref.session_id, limit=100,
                                 order=order))['result']['rows']
            assert Counter(row['row_kind'] for row in whole) == {'compaction': 1, 'message': 2}
            assert await _every_page(owner, ref.session_id, limit, order) == whole
        finally:
            await owner.close()
