"""A profile whose retirement misses its deadline keeps its slot (its writer may still run) but
refuses every admission: it must leave the published served set and the ticket store at once,
so discovery stops reporting it ready (and reads its parked verdict) instead of handing clients
a profile whose every submit answers runtime_draining."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway import run_runtime
from gateway.runtime_bootstrap import TicketStore
from gateway.session_authorities import SessionAuthorities
from gateway.session_contract import Submission
from tests.gateway.test_profile_retire_stops_turns import ACTOR, REF, _Agent, _authority


@pytest.mark.asyncio
async def test_still_retiring_profile_leaves_the_served_set(tmp_path, monkeypatch):
    from gateway import session_finite
    agent = _Agent(honours_stop=False)
    db, authority, execute, _returned = _authority(tmp_path, agent)
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    monkeypatch.setattr('gateway.session_cron.unbind_owner', lambda authority: None)
    launch = SimpleNamespace(profile_id=str(tmp_path / 'launch'), retiring=False)
    registry = SessionAuthorities(launch.profile_id)
    registry.add(launch.profile_id, launch)
    registry.add(authority.profile_id, authority, name='owned')
    runner = authority.runner
    runner.session_authorities = registry
    runner.session_runtime_descriptor = {'served_profiles': registry.served_profiles()}
    runner.session_ticket_store = TicketStore('owner', registry.profile_ids())
    try:
        with db:
            await authority.submit(ACTOR, Submission(request_id='running', ref=REF, payload={'text': 'a'}, intent='queue'))
            await asyncio.wait_for(asyncio.to_thread(agent.started.wait, 5), 6)
            assert await run_runtime._retire_profile_authority(authority, timeout=0.2) is False
            assert runner.session_runtime_descriptor['served_profiles'] == [
                {'profile_id': launch.profile_id, 'home': launch.profile_id}]
            with pytest.raises(PermissionError):
                runner.session_ticket_store.mint(profile_id='owned', subject='u', purpose='interactive')
    finally:
        agent.release.set()
        await asyncio.sleep(.05)
        db.close()
