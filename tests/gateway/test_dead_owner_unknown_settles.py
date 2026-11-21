"""A person's ``unknown`` turn whose owner process is provably gone settles as interrupted.

A person's lost turn pauses the session's FIFO so they can choose to resend it (GW-10). That
choice presumes somebody can still complete the fenced turn: a live owner holding a live turn
lease. When the conversation has lease rows and every one of them is expired or names a provably
dead pid, nobody can ever complete it — on a messaging surface the person may never see the
pause notice, and a Discord thread sat a full day behind one dead worker (Hermes 0.23.x,
2026-11-21) with every later message queued behind it. Such a head settles through the same
fenced resolution ``/discard`` uses: terminal ``interrupted``, never replayed, with the person's
failed-turn notice, and the queued followers run.

Without lease rows there is no proof the owner is gone, so the pause keeps its meaning: only a
provably dead (or expired) holder unlocks the FIFO. Real ``SessionStore`` SQLite, no store
monkeypatching.
"""
import os
import queue
import time
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_authority import LiveSession, initialize_session_authority
from gateway.session_contract import SessionRef, Submission
from gateway.session_controls import AuthorityConnection
from process_identity import holder_namespace_token, holder_process_is_dead
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.waiting import async_wait_until


class Peer:
    def __init__(self):
        self.frames = queue.Queue()

    def write(self, frame):
        self.frames.put(frame)
        return True


def _dead_holder() -> str:
    return f"pid=999999999{holder_namespace_token()}:turn=probe"  # a pid that cannot exist


def _live_holder() -> str:
    return f"pid={os.getpid()}{holder_namespace_token()}:turn=probe"


def _runner(tmp_path, monkeypatch, executed):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()['result'] = {'final_response': 'ACK_' + event.text, 'completed': True, 'messages': []}
        return 'ACK_' + event.text

    runner = SimpleNamespace(_session_db=store.persistence, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    return store, runner


async def _person_head_then_restart(tmp_path, monkeypatch, executed):
    """Owner 1 admits a person's head and a follower, claims the head, dies. Owner 2 starts on
    the same store. Returns (store, authority2, db, head_id, follower_id)."""
    store, runner = _runner(tmp_path, monkeypatch, executed)
    db = store.persistence
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='gw10')
    seed_session(db, 's', 'telegram')
    first = await initialize_session_authority(runner, profile_id='p', instance_id='owner-1')
    first.sessions['s'] = LiveSession(source, 's')
    ref = SessionRef('p', 's')
    actor = AuthorityConnection(first, Peer(), {'user_id': 'human'}).actor
    first._schedule = lambda ref: None
    receipt = await first.submit(actor, Submission('head', ref, {'text': 'HEAD'}, 'queue'))
    follower = await first.submit(actor, Submission('follower', ref, {'text': 'FOLLOWER'}, 'queue'))
    claimed = db.admissions.claim(epoch=first.epoch, session_id='s')
    assert claimed.admission_id == receipt.admission_id
    seed_messages(db, 's', {'role': 'user', 'content': 'HEAD'})
    second = await initialize_session_authority(runner, profile_id='p', instance_id='owner-2')
    second.sessions['s'] = LiveSession(source, 's')
    return store, second, db, receipt.admission_id, follower.admission_id


def _statuses(db):
    return {r.admission_id: (r.status, r.outcome) for r in db.admissions.list(session_id='s', pending_only=False)}


async def _run_drain(store, authority):
    authority._schedule(SessionRef('p', 's'))
    live = authority.sessions['s']
    await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain finished')


@pytest.mark.asyncio
async def test_dead_holder_lease_settles_the_persons_lost_turn_and_runs_the_fifo(tmp_path, monkeypatch):
    """A person's lost head behind a lease naming a provably dead pid settles interrupted; the
    queued follower runs without anybody typing ``/discard``."""
    executed = []
    store, authority, db, head_id, follower_id = await _person_head_then_restart(tmp_path, monkeypatch, executed)
    try:
        assert _statuses(db)[head_id] == ('unknown', None)
        assert holder_process_is_dead(_dead_holder()), 'probe sanity: the constructed holder must read as dead'
        assert db.coordination.try_acquire_turn_lease(session_id='s', holder=_dead_holder(), ttl_seconds=3600.0)
        leases = db.coordination.session_turn_leases(session_id='s')
        assert leases and leases[0][0] == _dead_holder(), leases
        await _run_drain(store, authority)
        assert executed == ['FOLLOWER']
        statuses = _statuses(db)
        assert statuses[head_id] == ('terminal', 'interrupted')
        assert statuses[follower_id] == ('terminal', 'completed')
        notices = [m['content'] for m in transcript_rows(db, 's') if m.get('display_kind') == 'failed_turn']
        assert notices, 'the settled turn must leave a failed-turn notice'
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_expired_lease_with_a_live_holder_still_settles(tmp_path, monkeypatch):
    """An expired lease cannot complete anything, even with a live holder: expiry itself is the
    authority (STRICT probe policy), so the head settles and the follower runs."""
    executed = []
    store, authority, db, head_id, follower_id = await _person_head_then_restart(tmp_path, monkeypatch, executed)
    try:
        # Expired lease, live holder: expiry itself is the authority (STRICT probe policy).
        live = _live_holder()
        assert not holder_process_is_dead(live), 'probe sanity: this process must read as alive'
        assert db.coordination.try_acquire_turn_lease(session_id='s', holder=live, ttl_seconds=0.2)
        time.sleep(0.3)
        await _run_drain(store, authority)
        assert executed == ['FOLLOWER']
        assert _statuses(db)[head_id] == ('terminal', 'interrupted')
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_live_lease_keeps_the_persons_pause(tmp_path, monkeypatch):
    """A live lease held by this very process keeps the person's pause exactly as before
    (GW-10 keeps its meaning)."""
    executed = []
    store, authority, db, head_id, follower_id = await _person_head_then_restart(tmp_path, monkeypatch, executed)
    try:
        assert db.coordination.try_acquire_turn_lease(session_id='s', holder=_live_holder(), ttl_seconds=3600.0)
        await _run_drain(store, authority)
        assert executed == []
        statuses = _statuses(db)
        assert statuses[head_id] == ('unknown', None)
        assert statuses[follower_id] == ('queued', None)
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_no_lease_rows_keeps_the_persons_pause(tmp_path, monkeypatch):
    """No lease rows = no proof the owner is gone: the person's lost turn still pauses the FIFO
    for them to resolve (``/discard``), exactly as GW-10 designed."""
    executed = []
    store, authority, db, head_id, follower_id = await _person_head_then_restart(tmp_path, monkeypatch, executed)
    try:
        assert db.coordination.session_turn_leases(session_id='s') == ()
        await _run_drain(store, authority)
        assert executed == []
        statuses = _statuses(db)
        assert statuses[head_id] == ('unknown', None)
        assert statuses[follower_id] == ('queued', None)
    finally:
        store.close_all_db_handles()
