"""A reused final keeps main's message.complete contract through the session authority.

The agent marks a final that repeats text the turn already delivered (``response_reused``);
the Desktop settles it in place instead of appending it after the bubble's last tool row.
"""
import asyncio
from types import SimpleNamespace

import pytest
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import SessionActor, SessionRef, Submission

ACTOR = SessionActor('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
REF = SessionRef('owned', 's')


@pytest.mark.asyncio
@pytest.mark.parametrize('reused', [True, False])
async def test_completion_carries_the_reused_final_flag(tmp_path, monkeypatch, reused):
    from gateway import session_finite
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = storage_handle(tmp_path)
    seed_session(db, 's', 'test')
    epoch = db.runtime.begin_epoch(instance_id='current')
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='current', epoch=epoch,
                                 persistence=db)
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
