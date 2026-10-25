"""A reused final keeps main's message.complete contract through the session authority.

The agent marks a final that repeats text the turn already delivered (``response_reused``);
the Desktop settles it in place instead of appending it after the bubble's last tool row.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import Principal, SessionRef, Submission
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch

ACTOR = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
REF = SessionRef('owned', 's')


@pytest.mark.asyncio
@pytest.mark.parametrize('reused', [True, False])
async def test_completion_carries_the_reused_final_flag(tmp_path, monkeypatch, reused):
    from gateway import session_finite
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = begin_runtime_epoch(db, instance_id='current')
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='current', db=db, epoch=epoch)
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
    frames = []
    authority.sessions['s'].event_stream.observers.add(frames.append)

    async def execute(authority, ref, row):
        authority.pending_results[row['admission_id']] = {'result': {
            'final_response': 'answered', 'messages': [], 'response_reused': reused,
            'response_transformed': False}, 'usage': {}}
        return 'answered'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        await authority.submit(ACTOR, Submission(request_id='r', ref=REF, payload={'text': 'q'}, intent='queue'))
        await asyncio.wait_for(authority.sessions['s'].task, 5)

    [complete] = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
    assert complete['text'] == 'answered'
    assert complete.get('response_reused', False) is reused
