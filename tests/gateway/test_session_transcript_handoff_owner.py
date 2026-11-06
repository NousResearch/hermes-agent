"""``session.transcript`` shows each compaction handoff once, before the first row in the whole transcript
that holds it, wherever a page starts: paged at any ``limit``, either order, the rows are exactly a full
read's, across identical uid-less handoffs on several carriers, several handoffs on one carrier, in-place
compactions and rotations.
"""
import random

import pytest

from tests.gateway.test_session_stored_reads import _call, _compact_carrying_tail, _summary
from tests.gateway.test_session_stored_reads_bounds import _session
from tests.gateway.test_session_transcript_snapshot import _every_page


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
    return rotations


@pytest.mark.asyncio
@pytest.mark.parametrize('seed', range(8))
async def test_every_limit_and_order_pages_exactly_the_full_read(tmp_path, monkeypatch, seed):
    with _session(tmp_path, monkeypatch) as (db, _authority, owner, ref, target):
        try:
            _seeded_transcript(db, target, seed)
            whole = (await _call(owner, 'session.transcript', session_id=ref.session_id, limit=500))['result']['rows']
            assert any(row['row_kind'] == 'compaction' for row in whole), whole
            for order in ('oldest', 'latest'):
                for limit in range(1, 6):
                    assert await _every_page(owner, ref.session_id, limit, order) == whole, (order, limit)
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
