"""Cancelling a queued admission settles every observer kind in the authority, not per caller."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import Principal, SessionRef, Submission
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch

ACTOR = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
REF = SessionRef('owned', 's')


def _authority(tmp_path, monkeypatch, platform=None):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = begin_runtime_epoch(db, instance_id='current')
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='current', db=db, epoch=epoch)
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=platform, user_id='human'), 'route')
    return db, authority


def _submit(authority, request_id, text=None):
    return authority.submit(ACTOR, Submission(request_id=request_id, ref=REF,
                                              payload={'text': text or request_id}, intent='queue'))


@pytest.mark.asyncio
async def test_cancel_queued_settles_native_waiter_and_publishes_terminal_completion(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    frames = []
    with db:
        receipt = await _submit(authority, 'queued-then-cancelled')
        # The native ingress delivery waiter (session_ingress.admit_message) and an
        # event-stream observer (ACP prompt(), API run projections) both wait on this row.
        authority.native_waiters.add(receipt.admission_id)
        waiter = authority.waiters.setdefault(receipt.admission_id, asyncio.get_running_loop().create_future())
        authority.sessions['s'].event_stream.observers.add(frames.append)

        cancelled = await authority.cancel_queued(ACTOR, REF, receipt.admission_id)
        assert (cancelled.status, cancelled.outcome) == ('terminal', 'cancelled')

        assert waiter.done() and receipt.admission_id not in authority.native_waiters
        assert receipt.admission_id not in authority.waiters
        terminal = [f['params'] for f in frames if f['params']['type'] == 'message.complete']
        assert len(terminal) == 1, [f['params']['type'] for f in frames]
        assert terminal[0]['admission_id'] == receipt.admission_id
        assert terminal[0]['payload']['outcome'] == 'cancelled'
        assert terminal[0]['payload']['admission_id'] == receipt.admission_id

        # Idempotent: a repeated cancel of the terminal row is not a second completion.
        await authority.cancel_queued(ACTOR, REF, receipt.admission_id)
        assert len([f for f in frames if f['params']['type'] == 'message.complete']) == 1

