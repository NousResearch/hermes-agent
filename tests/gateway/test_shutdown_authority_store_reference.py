"""A read through a session authority after the gateway's shutdown closes reopens its store and answers.

The stop's per-handle closes release the runner's handles on each profile's store, then ``close_all_stores`` closes
every store the process holds. Startup recovery can still read through an authority after that, when the stop lands
while the start is finishing. Such a read opens the store again with the ``hermes_state`` reopen warning, once per
store, and returns the store's answer.
"""
import logging
import time

import pytest

from tests.gateway import test_profile_reserve_reopen_warning as reopen

served = reopen.served

_WARNING = ("state.db connection for {} was closed while a read was still in flight — reopening "
            "(teardown/worker race, #94736)")


@pytest.mark.asyncio
async def test_authority_read_after_the_shutdown_closes_reopens_once_per_store(served, caplog):
    from gateway.run import GatewayRunner
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.run_shutdown import GatewayShutdownMixin
    from persistence_assembly import store_location
    runner, _alpha = served
    await initialize_gateway_runtime(runner)
    authorities = list(runner.session_authorities)
    assert len(authorities) == 2

    ctx = GatewayShutdownMixin._StopContext(deferred_count=lambda: 0, started_at=time.monotonic())
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        GatewayRunner._stop_quiesce_and_close_session_dbs(runner, 60.0, ctx)
    assert "SessionDB close done" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        for authority in authorities:
            assert authority.persistence.sessions.get("nope") is None
            assert authority.persistence.admissions.pending_targets(statuses=frozenset({"queued"})) == frozenset()
    assert sorted(reopen._reopen_warnings(caplog)) == sorted(
        _WARNING.format(store_location(authority.persistence.scope)) for authority in authorities)


@pytest.mark.asyncio
async def test_a_retired_authority_releases_its_store_reference(served):
    import persistence_assembly
    from gateway.run_runtime import initialize_gateway_runtime, unserve_profile_runtime
    runner, alpha = served
    await initialize_gateway_runtime(runner)
    key = runner.session_authorities.for_home(alpha.resolve()).persistence.scope.hermes_home_key
    held = persistence_assembly._shared[key][1]

    assert await unserve_profile_runtime(runner, alpha)
    assert persistence_assembly._shared[key][1] == held - 1
