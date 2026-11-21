"""prompt.move_queued: Desktop's reorder of a canonical queued row. A move changes only the order the
queued rows of one session run in: the drain claims the moved row in its new place, the running turn
and later admissions keep theirs, and every viewer is shown the new order."""
import asyncio

import pytest

from persistence.core.errors import AuthorityRefusal
from tests.gateway.test_prompt_queued_controls import Turn, _call, _connection, interactive
from tests.gateway.test_session_authority_cancel_settlement import _authority, _submit
from tests.waiting import async_wait_until


def _rows(authority):
    return authority.persistence.admissions.list(session_id='s')


def _queued(authority):
    return [row.request_id for row in _rows(authority) if row.status == 'queued']


async def _three_queued(authority):
    return {text: (await _submit(authority, text)).admission_id for text in ('a', 'b', 'c')}


@pytest.mark.asyncio
async def test_a_moved_row_is_claimed_in_its_new_place(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(authority.persistence)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        ids = await _three_queued(authority)
        conn = _connection(authority)

        top = await _call(conn, 'prompt.move_queued', admission_id=ids['c'], before_admission_id=ids['a'])
        assert 'error' not in top, top
        assert top['result']['status'] == 'queued'
        assert _queued(authority) == ['c', 'a', 'b']

        tail = await _call(conn, 'prompt.move_queued', admission_id=ids['c'], before_admission_id=None)
        assert 'error' not in tail, tail
        assert _queued(authority) == ['a', 'b', 'c']

        # The claim takes the head of the moved order, not the oldest admission.
        await _call(conn, 'prompt.move_queued', admission_id=ids['b'], before_admission_id=ids['a'])
        claimed = authority.persistence.admissions.claim(epoch=authority.epoch, session_id='s')
        assert claimed and claimed.request_id == 'b'

        # A later admission still lands at the back of the moved queue.
        await _submit(authority, 'd')
        assert _queued(authority) == ['a', 'c', 'd']


@pytest.mark.asyncio
async def test_move_publishes_the_new_order_to_viewers(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(authority.persistence)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    frames = []
    authority.sessions['s'].event_stream.observers.add(frames.append)
    with db:
        ids = await _three_queued(authority)
        frames.clear()
        await _call(_connection(authority), 'prompt.move_queued', admission_id=ids['c'], before_admission_id=ids['b'])
    pending = [f['params']['payload']['pending'] for f in frames
               if isinstance(f['params'].get('payload'), dict) and 'pending' in f['params']['payload']]
    assert pending, 'no pending snapshot published after the move'
    assert [row['input_id'] for row in pending[-1]] == ['a', 'c', 'b']
    assert all('move' in row['controls'] for row in pending[-1])


@pytest.mark.asyncio
async def test_a_running_turn_never_moves_and_its_queue_reorders_behind_it(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(authority.persistence)
    turn = Turn(monkeypatch)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(turn.claimed.wait(), 30)
        ids = {text: (await _submit(authority, text)).admission_id for text in ('x', 'y')}
        conn = _connection(authority)

        started = _rows(authority)[0]
        refused = await _call(conn, 'prompt.move_queued', admission_id=started.admission_id, before_admission_id=None)
        assert refused['error']['data']['reason'] == 'admission_started', refused

        moved = await _call(conn, 'prompt.move_queued', admission_id=ids['y'], before_admission_id=ids['x'])
        assert 'error' not in moved, moved
        assert [row.request_id for row in _rows(authority)] == ['running', 'y', 'x']

        turn.release.set()  # stays set: every later turn runs straight through
        await async_wait_until(lambda: len(turn.ran) == 3, timeout=30, what='the queue drained')
    assert turn.ran == ['running', 'y', 'x']


@pytest.mark.asyncio
async def test_move_refuses_an_anchor_that_is_not_queued_here(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(authority.persistence)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        ids = await _three_queued(authority)
        conn = _connection(authority)
        missing = await _call(conn, 'prompt.move_queued', admission_id=ids['a'], before_admission_id='nope')
        assert missing['error']['data']['reason'] == 'not_found', missing
        self_anchor = await _call(conn, 'prompt.move_queued', admission_id=ids['a'], before_admission_id=ids['a'])
        assert 'error' in self_anchor, self_anchor
        authority.persistence.admissions.claim(epoch=authority.epoch, session_id='s')  # 'a' starts
        gone = await _call(conn, 'prompt.move_queued', admission_id=ids['c'], before_admission_id=ids['a'])
        assert gone['error']['data']['reason'] == 'not_found', gone
    assert _queued(authority) == ['b', 'c']


@pytest.mark.asyncio
async def test_move_is_not_offered_on_a_non_interactive_session(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(authority.persistence, source='cron')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        ids = await _three_queued(authority)
        refused = await _call(_connection(authority), 'prompt.move_queued', admission_id=ids['c'],
                              before_admission_id=ids['a'])
        assert refused['error']['data']['reason'] == 'unsupported_control', refused
    assert _queued(authority) == ['a', 'b', 'c']


@pytest.mark.asyncio
async def test_the_store_move_only_permutes_the_queued_seqs(tmp_path, monkeypatch):
    """Atomic in the store: the queued rows trade their seq values, nothing else changes."""
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    admissions = authority.persistence.admissions
    with db:
        ids = await _three_queued(authority)
        before = {row.admission_id: row for row in _rows(authority)}
        order = admissions.move(epoch=authority.epoch, admission_id=ids['c'], before_admission_id=ids['a'])
        assert order == [ids['c'], ids['a'], ids['b']]
        after = {row.admission_id: row for row in _rows(authority)}
        assert sorted(row.seq for row in after.values()) == sorted(row.seq for row in before.values())
        assert {k: (v.request_id, v.payload, v.status) for k, v in after.items()} == {
            k: (v.request_id, v.payload, v.status) for k, v in before.items()}
        with pytest.raises(AuthorityRefusal):
            admissions.move(epoch=authority.epoch + 1, admission_id=ids['a'], before_admission_id=None)
