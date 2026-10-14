"""A gateway stop settles authority work within the watchdog leash it has left, and never closes
state.db under an authority turn that outlived the settle (N59).

The settle used to wait a flat TURN_SETTLE_SECONDS (45 s) whatever the drain and interrupt phases had
already spent, so the stop ran past its watchdog leash; and the SessionDB close gate counted executor,
cron, API and deferred workers but not the authority's still-running turn, so it closed and
checkpointed state.db under a live writer.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

import gateway.run as gw_mod
from gateway import run_shutdown
from tests.gateway.test_shutdown_executor_quiesce import _FakeGateway


@pytest.mark.asyncio
async def test_stop_settles_within_the_leash_and_keeps_state_db_open_for_a_live_turn(monkeypatch):
    events = []
    gw = _FakeGateway(events)
    # A 12 s watchdog leash: the settle may spend at most what is left of it minus the tail reserve.
    monkeypatch.setattr(run_shutdown, "resolve_shutdown_watchdog_delay", lambda drain, **_k: 12.0)
    never = asyncio.Event()
    turn = asyncio.create_task(never.wait())  # an executing turn that ignores its Stop
    live = SimpleNamespace(task=turn, route="r", event_stream=SimpleNamespace(execution={"execution_generation": 1}))
    gw.session_authority = SimpleNamespace(
        profile_id="default", sessions={"s": live}, _managed_workers={}, adopted={}, pending_stops={},
        runner=gw)
    started = time.monotonic()
    try:
        async with asyncio.timeout(10):
            await gw_mod.GatewayRunner.stop(gw)
    finally:
        turn.cancel()
    assert time.monotonic() - started < 6, "the settle overran the stop's watchdog leash"
    assert "close:session_db" not in events, f"state.db closed under a live authority turn: {events}"
