"""``session.transcript`` forward polling: a poller that keeps the oldest-order ``next_cursor`` never silently
misses or duplicates a row. Appends arrive as new rows (``reset: false``); an in-place compaction, a
rotation, a rewind or a rewrite of a held row between polls answers ``reset: true`` from the first page.

The reader below is the documented one: append each page's rows, drop what it holds on ``reset``, stop at
an empty page and keep its cursor. After every change it must hold exactly what a fresh read shows.
"""
import random

import pytest

from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import ConversationRequest, NewMessage, RewindTargetUnavailable
from tests.fixtures.storage_read import conversation
from tests.gateway.test_session_stored_reads import (
    _append, _call, _compact_carrying_tail, _legacy_owner, _native, _summary,
)


def _shape(rows):
    return [(row['row_kind'], row.get('role'), row.get('row_id'), row.get('text'), row.get('display_metadata'))
            for row in rows]


class Reader:
    def __init__(self, connection, session_id, limit=2):
        self.connection, self.session_id, self.limit = connection, session_id, limit
        self.rows, self.cursor, self.resets = [], None, []

    async def poll(self):
        """Read until caught up; returns the rows this poll added (all of them after a reset)."""
        added = []
        while True:
            reply = await _call(self.connection, 'session.transcript', session_id=self.session_id, limit=self.limit,
                                **({'cursor': self.cursor} if self.cursor else {}))
            assert 'result' in reply, reply
            result = reply['result']
            self.resets.append(result['reset'])
            if result['reset']:
                self.rows, added = [], []
            self.rows += result['rows']
            added += result['rows']
            assert result['next_cursor'], result  # oldest order: always a cursor to poll from
            self.cursor = result['next_cursor']
            if not result['rows']:
                return added

    async def fresh(self):
        return _shape(await Reader(self.connection, self.session_id, self.limit).poll())


def _rotate(db, parent, child, *messages):
    """A compression rotation of *parent* into *child* carrying *messages* (live dicts)."""
    db.lineage.publish_compression_child(parent, CompressionChild(session_id=child, source='tui', model='m'),
                                         [NewMessage.from_live(message) for message in messages],
                                         require_compression_lease=False)


@pytest.mark.asyncio
async def test_a_poller_gets_appends_as_new_rows_and_resets_on_every_other_change(tmp_path, monkeypatch):
    from gateway.session_local_recovery import transcript_target
    from tests.gateway.test_session_mutation_model_busy import _owner

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        target = transcript_target(authority, ref)
        reader = Reader(owner, ref.session_id)
        assert await reader.poll() == [] and reader.resets == [True]  # empty transcript: a cursor all the same
        for i in (1, 2, 3):
            _append(db, target, 'user', f'question {i}')
            _append(db, target, 'assistant', f'answer {i}')
        reader.resets.clear()
        assert [row['text'] for row in await reader.poll()] == [
            'question 1', 'answer 1', 'question 2', 'answer 2', 'question 3', 'answer 3']
        assert not any(reader.resets), reader.resets
        assert _shape(reader.rows) == await reader.fresh()

        # Append: only the new row, no reset.
        _append(db, target, 'user', 'question 4')
        reader.resets.clear()
        assert [row['text'] for row in await reader.poll()] == ['question 4'] and not any(reader.resets)

        # In-place compaction: start over; the handoff is a row at its position.
        _compact_carrying_tail(db, target, _summary('## Goal\nshrink', 6), tail_rows=1)
        reader.resets.clear()
        await reader.poll()
        assert reader.resets[0] is True and not any(reader.resets[1:]), reader.resets
        assert ('compaction', '## Goal\nshrink') in [(row['row_kind'], row.get('text')) for row in reader.rows]
        assert _shape(reader.rows) == await reader.fresh()

        # A rewrite of a held row (a reaction): start over, and the reader now holds the reacted row.
        held = next(row['row_id'] for row in reader.rows if row.get('text') == 'question 4')
        db.messages.set_reaction(target, held, '\U0001f44d', author='user')
        reader.resets.clear()
        await reader.poll()
        assert reader.resets[0] is True, reader.resets
        assert _shape(reader.rows) == await reader.fresh()

        # A rewind of held rows: start over, the rewound rows gone.
        _append(db, target, 'assistant', 'answer 4')
        await reader.poll()
        db.messages.rewind_user_turn(target, -1)  # the newest user turn: the held 'question 4'
        reader.resets.clear()
        await reader.poll()
        assert reader.resets[0] is True, reader.resets
        assert 'answer 4' not in [row.get('text') for row in reader.rows]
        assert _shape(reader.rows) == await reader.fresh()
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_rows_written_and_compacted_between_two_polls_reset_the_poller(tmp_path, monkeypatch):
    """The rows the cursor never covered were appended and compacted before the next poll: the
    compaction is a non-append change, so the poll starts over instead of passing their copies as new."""
    from gateway.session_local_recovery import transcript_target
    from tests.gateway.test_session_mutation_model_busy import _owner

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        target = transcript_target(authority, ref)
        _append(db, target, 'user', 'question 1')
        reader = Reader(owner, ref.session_id)
        await reader.poll()
        _append(db, target, 'assistant', 'answer 1')
        _append(db, target, 'user', 'question 2')
        _compact_carrying_tail(db, target, _summary('## Goal\nboth', 3), tail_rows=1)
        reader.resets.clear()
        await reader.poll()
        assert reader.resets[0] is True, reader.resets
        texts = [row.get('text') for row in reader.rows]
        assert texts == ['question 1', 'answer 1', '## Goal\nboth', 'question 2'], texts
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_rotation_between_polls_resets_the_poller_onto_the_new_segment(tmp_path, monkeypatch):
    from gateway.session_authority import initialize_session_authority

    store, db, runner = _legacy_owner(tmp_path, monkeypatch)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = _native(authority, tmp_path, 'native')
    try:
        reader = Reader(native, 'old-tui')
        await reader.poll()
        assert [row.get('text') for row in reader.rows] == [
            'before compression', 'old answer', 'after compression', 'new answer']
        _append(db, 'tui-tip', 'user', 'tip question')
        reader.resets.clear()
        assert [row.get('text') for row in await reader.poll()] == ['tip question'] and not any(reader.resets)
        _rotate(db, 'tui-tip', 'tui-tip-2', {'role': 'user', 'content': 'rotated question'})
        reader.resets.clear()
        await reader.poll()
        assert reader.resets[0] is True, reader.resets
        assert [row.get('text') for row in reader.rows][-1] == 'rotated question'
        assert _shape(reader.rows) == await reader.fresh()
    finally:
        await native.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_poller_matches_a_fresh_read_after_any_sequence_of_changes(tmp_path, monkeypatch):
    """Seeded random appends, compactions, rewinds and reactions between polls: after every poll the
    reader holds exactly the rows a fresh read shows (nothing missed, nothing twice)."""
    from gateway.session_local_recovery import transcript_target
    from tests.gateway.test_session_mutation_model_busy import _owner

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        target = transcript_target(authority, ref)
        rng, n = random.Random(150102), [0]
        reader = Reader(owner, ref.session_id, limit=3)

        def append():
            n[0] += 1
            _append(db, target, 'user' if n[0] % 2 else 'assistant', f'message {n[0]}')

        def live_ids():
            return [row['_row_id'] for row in conversation(db, target, ConversationRequest(include_row_ids=True))]
        for step in range(40):
            op = rng.choice(['append', 'append', 'append', 'compact', 'rewind', 'react'])
            ids = live_ids()
            if op == 'append' or len(ids) < 4:
                for _ in range(rng.randint(1, 3)):
                    append()
            elif op == 'compact':
                _compact_carrying_tail(db, target, _summary(f'## Goal\nstep {step}', len(ids)), tail_rows=2)
            elif op == 'rewind':
                try:
                    db.messages.rewind_user_turn(target, -1)
                except RewindTargetUnavailable:
                    pass  # no user turn left to rewind
            else:
                db.messages.set_reaction(target, rng.choice(ids), rng.choice(['\U0001f44d', '\U0001f525']), author='user')
            await reader.poll()
            assert _shape(reader.rows) == await reader.fresh(), (step, op)
    finally:
        await owner.close()
        store.close_all_db_handles()
