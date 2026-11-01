"""Queue fairness on the owner's FIFO: a full session cap queues instead of refusing (people before
automation), each queued row says why it waits, an edit cannot start while open (and keeps its place
when the profile opts in), and queued rows can be reordered."""
import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gateway.session_queue_holds import _slot_blocked, set_edit_hold
from hermes_state_runtime import list_session_admissions
from tests.waiting import async_wait_until
from tests.gateway.test_prompt_queued_controls import Turn, _call, _connection, interactive
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


def _queued(db):
    return [r['payload']['text'] for r in list_session_admissions(db, session_id='s') if r['status'] == 'queued']


def _reasons(authority):
    return {row['input_id']: row.get('hold_reason') for row in (
        asdict(authority._pending_receipt(r)) for r in authority._pending_rows(REF))}


@pytest.mark.asyncio
async def test_an_opted_in_edit_keeps_its_place_and_a_reorder_moves_rows(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    (tmp_path / 'config.yaml').write_text('display:\n  queue_edit_keeps_place: true\n', encoding='utf-8')
    interactive(db)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        await _submit(authority, 'one')
        two = await _submit(authority, 'two')
        three = await _submit(authority, 'three')
        conn = _connection(authority)

        edited = await _call(conn, 'prompt.replace_queued', admission_id=two.admission_id,
                             submission_id='two-edited', text='two, edited')
        assert 'error' not in edited, edited
        assert _queued(db) == ['one', 'two, edited', 'three'], 'the edit must keep its FIFO place'

        moved = await _call(conn, 'prompt.reorder_queued', admission_id=three.admission_id, position=0)
        assert 'error' not in moved, moved
        assert _queued(db) == ['three', 'one', 'two, edited']
        assert moved['result']['order'][0] == three.admission_id

        last = await _call(conn, 'prompt.reorder_queued', admission_id=three.admission_id, position=99)
        assert 'error' not in last, last
        assert _queued(db) == ['one', 'two, edited', 'three']


@pytest.mark.asyncio
async def test_a_held_head_never_starts_until_its_edit_ends(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    turn = Turn(monkeypatch)
    real, drains = authority._schedule, {'on': False}
    monkeypatch.setattr(authority, '_schedule', lambda ref: real(ref) if drains['on'] else None)
    with db:
        conn = _connection(authority)
        head = await _submit(authority, 'head')
        held = await _call(conn, 'prompt.hold_queued', admission_id=head.admission_id, held=True)
        assert 'error' not in held, held
        assert _reasons(authority)['head'] == 'editing'
        drains['on'] = True
        authority._schedule(REF)
        # The drain reached the held head and parked on it instead of starting it.
        await async_wait_until(lambda: authority.hold_reasons.get('s') == 'editing', timeout=30)
        assert turn.ran == [], 'a held admission started mid-edit'

        released = await _call(conn, 'prompt.hold_queued', admission_id=head.admission_id, held=False)
        assert 'error' not in released, released
        await asyncio.wait_for(turn.claimed.wait(), 30)
        turn.release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 30)
    assert turn.ran == ['head']


@pytest.mark.asyncio
async def test_a_full_session_cap_queues_the_turn_and_names_why(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    running = {'count': 1}
    authority.runner.config.max_concurrent_sessions = 1
    authority.runner._is_session_running = lambda route: False
    authority.runner._running_agent_count = lambda: running['count']
    turn = Turn(monkeypatch)
    with db:
        await _submit(authority, 'waits for a slot')
        await async_wait_until(lambda: _reasons(authority).get('waits for a slot') == 'slot', timeout=30)
        assert turn.ran == [], 'the head started over a full session cap'
        running['count'] = 0  # another session finished
        await asyncio.wait_for(turn.claimed.wait(), 30)
        turn.release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 30)
    assert turn.ran == ['waits for a slot']
    assert not _reasons(authority)


@pytest.mark.asyncio
async def test_a_finished_turn_frees_the_slot_it_won(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    authority.runner.config.max_concurrent_sessions = 1
    authority.runner._is_session_running = lambda route: False
    authority.runner._running_agent_count = lambda: 0
    turn = Turn(monkeypatch)
    with db:
        await _submit(authority, 'short turn')
        await asyncio.wait_for(turn.claimed.wait(), 30)
        turn.release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 30)
    # No other drain polled while it ran: the idle gateway's one slot is free for the next session.
    other = SimpleNamespace(route='other', source=SimpleNamespace(platform=None))
    row = {'principal_id': 'messaging:x', 'payload': {'text': 'hi'}}
    assert not _slot_blocked(authority, SimpleNamespace(session_id='other'), other, row)


def test_a_freed_slot_goes_to_a_person_before_automation():
    runner = SimpleNamespace(config=SimpleNamespace(max_concurrent_sessions=1),
                             _is_session_running=lambda route: False, _running_agent_count=lambda: 0)
    authority = SimpleNamespace(runner=runner, profile_id='p')
    live = SimpleNamespace(route='r', source=SimpleNamespace(platform=None))
    cron = {'principal_id': 'cron-owner', 'payload': {'text': 'tick'}}
    person = {'principal_id': 'messaging:x', 'payload': {'text': 'hi'}}
    # The cron head waited first, but a person is also waiting: the person takes the one free slot.
    runner._running_agent_count = lambda: 1
    assert _slot_blocked(authority, SimpleNamespace(session_id='cron'), live, cron)
    assert _slot_blocked(authority, SimpleNamespace(session_id='chat'), live, person)
    runner._running_agent_count = lambda: 0
    assert _slot_blocked(authority, SimpleNamespace(session_id='cron'), live, cron)
    assert not _slot_blocked(authority, SimpleNamespace(session_id='chat'), live, person)
    assert _slot_blocked(authority, SimpleNamespace(session_id='cron'), live, cron), "the person's won slot is reserved"


def test_an_edit_hold_lapses(monkeypatch):
    from gateway import session_queue_holds
    authority = SimpleNamespace(edit_holds={})
    set_edit_hold(authority, 'a', True)
    assert session_queue_holds.edit_held(authority, 'a')
    monkeypatch.setattr(session_queue_holds.time, 'monotonic', lambda: 10 ** 9)
    assert not session_queue_holds.edit_held(authority, 'a'), 'a crashed editor pinned the FIFO'


def _slot_authority(sources=None):
    runner = SimpleNamespace(config=SimpleNamespace(max_concurrent_sessions=1), _running_agent_count=lambda: 0)
    running = set()
    runner._is_session_running = lambda route: route in running
    return SimpleNamespace(runner=runner, profile_id='p', launch_sources=sources or {}), running


def test_a_won_slot_stays_reserved_until_its_turn_runs():
    authority, running = _slot_authority()
    live = lambda sid: SimpleNamespace(route=sid, source=SimpleNamespace(platform=None))
    row = {'principal_id': 'messaging:x', 'payload': {'text': 'hi'}}
    assert not _slot_blocked(authority, SimpleNamespace(session_id='one'), live('one'), row)
    # 'one' is still claiming (its turn is not counted as running yet): the one slot is taken.
    assert _slot_blocked(authority, SimpleNamespace(session_id='two'), live('two'), row), 'two drains took one slot'
    running.add('one')
    authority.runner._running_agent_count = lambda: 1
    assert _slot_blocked(authority, SimpleNamespace(session_id='two'), live('two'), row)
    running.clear()
    authority.runner._running_agent_count = lambda: 0
    assert not _slot_blocked(authority, SimpleNamespace(session_id='two'), live('two'), row)


def test_a_forwarded_a2a_turn_yields_a_freed_slot_to_a_person():
    from gateway.config import Platform
    authority, _ = _slot_authority({'a2a-s': 'a2a', 'chat': 'tui'})
    live = lambda sid: SimpleNamespace(route=sid, source=SimpleNamespace(platform=Platform.LOCAL))
    # A forward submits as the hosting local principal with plain text: only the launch source says a2a.
    forwarded = {'principal_id': 'local:owner', 'payload': {'text': 'peer asks'}, 'target_session_id': 'a2a-s'}
    person = {'principal_id': 'local:owner', 'payload': {'text': 'hi'}, 'target_session_id': 'chat'}
    authority.runner._running_agent_count = lambda: 1
    assert _slot_blocked(authority, SimpleNamespace(session_id='a2a-s'), live('a2a-s'), forwarded)
    assert _slot_blocked(authority, SimpleNamespace(session_id='chat'), live('chat'), person)
    authority.runner._running_agent_count = lambda: 0
    assert _slot_blocked(authority, SimpleNamespace(session_id='a2a-s'), live('a2a-s'), forwarded)
    assert not _slot_blocked(authority, SimpleNamespace(session_id='chat'), live('chat'), person)


@pytest.mark.asyncio
async def test_one_editor_closing_never_releases_another_editors_hold(tmp_path, monkeypatch):
    from gateway.session_queue_holds import edit_held
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        head = await _submit(authority, 'head')
        first, second = _connection(authority), _connection(authority)
        for conn in (first, second):
            assert 'error' not in await _call(conn, 'prompt.hold_queued', admission_id=head.admission_id, held=True)
        assert 'error' not in await _call(first, 'prompt.hold_queued', admission_id=head.admission_id, held=False)
        assert edit_held(authority, head.admission_id), "the first editor released the second editor's hold"


@pytest.mark.asyncio
async def test_a_reconnected_editor_releases_its_own_hold(tmp_path, monkeypatch):
    from gateway.session_queue_holds import edit_held
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        head = await _submit(authority, 'head')
        before, after = _connection(authority), _connection(authority)  # the same editor, a new socket
        for token in ('mine', 'other window'):
            assert 'error' not in await _call(before, 'prompt.hold_queued', admission_id=head.admission_id,
                                              held=True, holder=token)
        assert 'error' not in await _call(after, 'prompt.hold_queued', admission_id=head.admission_id,
                                          held=False, holder='mine')
        assert edit_held(authority, head.admission_id), "a release ended another window's hold"
        assert 'error' not in await _call(after, 'prompt.hold_queued', admission_id=head.admission_id,
                                          held=False, holder='other window')
        assert not edit_held(authority, head.admission_id), 'a cancelled edit kept the queue waiting'


@pytest.mark.asyncio
async def test_a_hold_waits_for_a_claim_in_flight_and_then_refuses(tmp_path, monkeypatch):
    from hermes_state_runtime import claim_session_input
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        head = await _submit(authority, 'head')
        gate = authority.sessions['s'].claim_gate
        await gate.acquire()  # a drain between its head check and its claim commit
        hold = asyncio.ensure_future(_call(_connection(authority), 'prompt.hold_queued',
                                           admission_id=head.admission_id, held=True))
        for _ in range(20):  # let the hold reach the gate
            await asyncio.sleep(0)
        claim_session_input(db, epoch=authority.epoch, session_id='s')
        gate.release()
        answer = await asyncio.wait_for(hold, 30)
    assert 'error' in answer, 'a hold was accepted on a row the drain had already claimed'
