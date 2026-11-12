"""Unserve releases a profile's gateway.lock only after its handles in that home close (N97)."""
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_unserve_holds_the_profile_lock_until_its_handles_close(tmp_path, monkeypatch):
    from gateway.run_profile_reconcile import GatewayProfileReconcileMixin
    from gateway.runtime_ownership import OwnershipConflict, exclusive_maintenance, process_ownership

    home = tmp_path / 'gone'
    process_ownership.reserve([home])
    deleter_could_start = []

    def close_all_under(directory):
        # A `hermes profile delete` waiting on the lock rmtree's the home the moment it wins.
        try:
            with exclusive_maintenance([home]):
                deleter_could_start.append(True)
        except OwnershipConflict:
            deleter_could_start.append(False)
        return 0

    monkeypatch.setattr('hermes_state_registry.close_all_under', close_all_under)
    runner = SimpleNamespace(
        _profile_failed_platforms={}, _profile_adapters={}, _served_profile_homes={'gone': home},
        _served_profile_signatures={}, _agent_cache={}, _evict_cached_agent=lambda key: None,
        config=SimpleNamespace(_runtime_profile_homes=(('gone', str(home)),)))
    try:
        await GatewayProfileReconcileMixin._unserve_profile(runner, 'gone', home)
        assert deleter_could_start == [False], 'gateway.lock released while store handles were open'
        assert not process_ownership.owns(home)
    finally:
        process_ownership.release(home)
