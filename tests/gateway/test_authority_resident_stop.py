"""Authority Stop must reach a running agent after its reusable cache entry is evicted."""
from types import SimpleNamespace

import pytest

from hermes_state_runtime import admit_session_input, claim_session_input
from tests.gateway.test_reaped_eviction_interrupts_run import _build_gateway, KEY
from tests.gateway.test_session_authority_cancel_settlement import _authority, ACTOR, REF


@pytest.mark.asyncio
async def test_stop_reaches_resident_agent_after_real_cache_eviction(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    calls = []
    agent = SimpleNamespace(interrupt=lambda: calls.append('stop'))
    runner, _state = _build_gateway(agent, [])
    authority.runner = runner
    authority.sessions['s'].route = KEY
    with db:
        admit_session_input(db, epoch=authority.epoch, principal_id=ACTOR.subject, session_id='s',
                            request_id='running', payload={'text':'work'})
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        runner._evict_cached_agent(KEY)
        assert runner._cached_agent_for(KEY) is None
        assert runner._resident_agent_for(KEY) is agent
        await authority.interrupt(ACTOR, REF, row['generation'])
        assert calls == ['stop']
        assert not authority.pending_stops
