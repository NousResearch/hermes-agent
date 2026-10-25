"""A claim abandoned inside THIS owner's epoch is fenced ``unknown`` by the drain.

Boot recovery stamps ``started`` rows of an *older* epoch only (``recover_session_inputs``), and
``/discard`` resolves ``unknown`` rows only. So a turn whose execution dies in the running
owner's own epoch — a lost managed worker on a person's message, a drain cancelled by a shutdown
— used to stay ``started`` for good: the FIFO never moved, ``/discard`` answered "nothing to
discard", and the pause notice named a command that did nothing. The drain's exit must fence the
claim it abandons, in the same epoch, so the advertised way out works.
"""
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_authority import LiveSession, initialize_session_authority
from gateway.session_contract import SessionRef, Submission
from gateway.session_controls import AuthorityConnection
from gateway.session_managed_worker import ManagedExecutionUnknown
from gateway.slash_commands_session import discard_lost_turns
from hermes_state_runtime import list_session_admissions
from tests.waiting import async_wait_until


class Peer:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)
        return True


def _statuses(db):
    return {r['admission_id']: (r['status'], r['outcome'])
            for r in list_session_admissions(db, session_id='s', pending_only=False)}


@pytest.mark.asyncio
async def test_a_drain_that_abandons_a_persons_turn_fences_it_so_discard_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='gw-abandon')
    db.create_session('s', source='telegram')

    async def answer(event):
        return 'ACK_' + event.text

    runner = SimpleNamespace(_session_db=db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner-1')
    authority.sessions['s'] = LiveSession(source, 's')
    ref = SessionRef('p', 's')
    actor = AuthorityConnection(authority, Peer(), {'user_id': 'human'}).actor
    ran = []
    try:
        authority._schedule = lambda ref: None
        head = await authority.submit(actor, Submission('head', ref, {'text': 'HEAD'}, 'queue'))
        follower = await authority.submit(actor, Submission('follower', ref, {'text': 'FOLLOWER'}, 'queue'))

        async def execute(owner, ref, row):
            if row['admission_id'] == head.admission_id:
                # A lost managed worker that committed no verdict: the person's turn is abandoned
                # in this owner's epoch, the case no restart and no ``/discard`` can reach.
                raise ManagedExecutionUnknown('managed_worker_unknown')
            ran.append(row['payload']['text'])
            return 'ACK'

        monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
        del authority._schedule
        authority._schedule(ref)
        live = authority.sessions['s']
        await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain paused')
        assert ran == []
        assert _statuses(db)[head.admission_id] == ('unknown', None)

        reply = await discard_lost_turns(authority, 's')
        assert '1' in reply, reply
        await async_wait_until(lambda: ran == ['FOLLOWER'], what='follower ran after discard')
        await async_wait_until(lambda: _statuses(db)[follower.admission_id][0] == 'terminal',
                               what='follower settled')
        assert _statuses(db)[head.admission_id][0] == 'terminal'
    finally:
        store.close_all_db_handles()
