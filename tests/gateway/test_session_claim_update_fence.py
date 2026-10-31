"""A turn that arrives while an in-app update holds the process fence waits queued, never starts."""
import asyncio

import pytest

from gateway import session_finite
from hermes_state_runtime import list_session_admissions
from tests.gateway.test_session_authority_cancel_settlement import _authority, _submit
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_a_turn_admitted_during_the_update_permit_waits_and_runs_once_it_ends(tmp_path, monkeypatch):
    from hermes_cli import backend_retirement, web_server_idle_proof

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, 'retirement', fence)
    monkeypatch.setattr(web_server_idle_proof, 'idle_proof', lambda **_: {'idle': True})
    db, authority = _authority(tmp_path, monkeypatch)
    ran = []

    async def execute(authority, ref, row):
        ran.append(row['payload']['text'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    permit = fence.prepare()
    assert permit['ok'] is True
    refused, acquire = [], fence.acquire

    def counted_acquire():
        admitted = acquire()
        refused.extend([] if admitted else [1])
        return admitted
    monkeypatch.setattr(fence, 'acquire', counted_acquire)
    with db:
        await _submit(authority, 'sent-mid-update')
        # The drain tried to claim (and was held) more than once: it is parked, not gone.
        await async_wait_until(lambda: len(refused) >= 2, timeout=30, what='the drain parks on the fence')
        # The update proved idle and holds the permit: the message is saved, its turn has not started.
        assert ran == []
        assert [row['status'] for row in list_session_admissions(db, session_id='s')] == ['queued']
        assert fence.cancel(permit['token']) == {'ok': True}
        await async_wait_until(lambda: ran == ['sent-mid-update'], timeout=30, what='the held turn runs')
        await asyncio.wait_for(authority.sessions['s'].task, 30)
    assert fence.active_count() == 0
