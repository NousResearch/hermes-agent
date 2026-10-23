"""A preclaim ``admission_conflict`` settles that head instead of pausing its FIFO for good."""
import asyncio

import pytest

from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


@pytest.mark.asyncio
async def test_conflicted_head_settles_with_its_refusal_and_the_successor_runs(tmp_path, monkeypatch):
    from gateway import session_api_turn, session_finite
    from gateway.config import Platform

    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.API_SERVER)
    executed = []

    def check_api_turn(authority, ref, payload):
        if payload['text'] == 'stale':
            raise RuntimeStoreError('admission_conflict')  # its route moved under it, on every drain
    monkeypatch.setattr(session_api_turn, 'check_api_turn', check_api_turn)

    async def execute(authority, ref, row):
        executed.append(row['payload']['text'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        authority._schedule = lambda ref: None
        head = await _submit(authority, 'stale')
        await _submit(authority, 'follower')
        waiter = authority.waiters.setdefault(head.admission_id, asyncio.get_running_loop().create_future())
        del authority._schedule
        authority._schedule(REF)
        await asyncio.wait_for(authority.sessions['s'].task, 5)

        assert executed == ['follower']
        statuses = {r['request_id']: (r['status'], r['outcome'])
                    for r in list_session_admissions(db, session_id='s', pending_only=False)}
        assert statuses == {'stale': ('terminal', 'cancelled'), 'follower': ('terminal', 'completed')}
        assert waiter.exception().reason == 'admission_conflict'
