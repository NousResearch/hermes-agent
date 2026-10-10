"""Unserving a profile stops and joins its running turns before ownership is released.

Cancelling the drain task alone leaves the turn's executor thread calling tools and writing
history after ``gateway.lock`` and the store handles are released. Retirement must Stop the
turn, wait for its thread (the admission settles), claim no successor, and report a turn that
outlives the deadline so the caller keeps the reservation."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import run_runtime
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import Principal, SessionRef, Submission
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, list_session_admissions

ACTOR = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
REF = SessionRef('owned', 's')


class _Agent:
    """Blocks its executor thread until interrupted (or released), like a turn in a model call."""

    def __init__(self, honours_stop=True):
        self.honours_stop = honours_stop
        self.stopped = threading.Event()
        self.release = threading.Event()
        self.started = threading.Event()
        self.writes_after_return = []

    def hard_interrupt(self, message=None, *, tool_reason=None):
        if self.honours_stop:
            self.stopped.set()

    def run(self):
        self.started.set()
        while not (self.stopped.is_set() or self.release.is_set()):
            self.stopped.wait(.01)
        return {'final_response': '', 'interrupted': True, 'messages': []}


def _authority(tmp_path, agent):
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = begin_runtime_epoch(db, instance_id='owner')
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None, _running_agents={})
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db, epoch=epoch)
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
    returned = threading.Event()

    async def execute(authority, ref, row):
        # The in-process turn: its agent is registered for the route and runs in an executor thread.
        runner._running_agents['route'] = agent
        try:
            result = await asyncio.get_running_loop().run_in_executor(None, agent.run)
        finally:
            runner._running_agents.pop('route', None)
        returned.set()
        authority.pending_results[row['admission_id']] = {'result': result, 'usage': {}}
        return ''
    return db, authority, execute, returned


@pytest.mark.asyncio
@pytest.mark.parametrize('honours_stop', [True, False])
async def test_unserve_stops_and_joins_the_running_turn_before_release(tmp_path, monkeypatch, honours_stop):
    from gateway import session_finite
    agent = _Agent(honours_stop=honours_stop)
    db, authority, execute, returned = _authority(tmp_path, agent)
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    monkeypatch.setattr(run_runtime, 'TURN_SETTLE_SECONDS', 1.0, raising=False)
    monkeypatch.setattr('gateway.session_cron.unbind_owner', lambda authority: None)
    try:
        with db:
            await authority.submit(ACTOR, Submission(request_id='running', ref=REF, payload={'text': 'a'}, intent='queue'))
            await authority.submit(ACTOR, Submission(request_id='follower', ref=REF, payload={'text': 'b'}, intent='queue'))
            await asyncio.wait_for(asyncio.to_thread(agent.started.wait, 5), 6)

            retired = await run_runtime._retire_profile_authority(authority)

            # Nothing of the old turn may run once retirement reports the profile releasable.
            if honours_stop:
                assert returned.is_set() and agent.stopped.is_set(), 'turn thread outlived retirement'
                assert retired is True
                rows = {r['request_id']: r for r in list_session_admissions(db, session_id='s', pending_only=False)}
                assert rows['running']['status'] == 'terminal' and rows['running']['outcome'] == 'interrupted'
                assert rows['follower']['status'] == 'queued', rows  # no successor claimed after Stop
            else:
                # The thread is still in its call: the caller must keep gateway.lock and the store.
                assert retired is False and not returned.is_set()
    finally:
        agent.release.set()
        await asyncio.sleep(.05)
        db.close()


class _RoomService:
    """Like the hosted runtime's stop: never interrupts the accepted member turn, and its room
    thread cannot finish until that turn has returned."""

    def __init__(self, returned):
        self.returned = returned

    def stop(self, *, timeout):
        return self.returned.wait(min(timeout, 1.5))


@pytest.mark.asyncio
async def test_retire_stops_the_turn_a_hosted_room_stop_waits_for(tmp_path, monkeypatch):
    from gateway import session_finite
    agent = _Agent()
    db, authority, execute, returned = _authority(tmp_path, agent)
    authority.hosted_room_service = _RoomService(returned)
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    monkeypatch.setattr(run_runtime, 'TURN_SETTLE_SECONDS', 1.0, raising=False)
    monkeypatch.setattr('gateway.session_cron.unbind_owner', lambda authority: None)
    try:
        with db:
            await authority.submit(ACTOR, Submission(request_id='running', ref=REF, payload={'text': 'a'}, intent='queue'))
            await asyncio.wait_for(asyncio.to_thread(agent.started.wait, 5), 6)

            retired = await run_runtime._retire_profile_authority(authority)

            assert agent.stopped.is_set(), 'the room stop failed before the turn was ever told to stop'
            assert retired is True and returned.is_set()
    finally:
        agent.release.set()
        await asyncio.sleep(.05)
        db.close()
