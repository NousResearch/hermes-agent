"""Regression: relay transport teardown must drain in-flight outbound frames.

Staging incident 2026-08-09 (frozen preview "It launched but ▉"): a trailing
finalize edit was in flight — awaiting its ``outbound_result`` — when the
gateway tore down the relay transport. ``disconnect()`` failed the pending
future immediately with "relay transport closed", so the finalize edit never
reached the platform even though the connector socket was still perfectly
able to serve it.

Contract under test: ``disconnect()`` gives in-flight outbound requests a
bounded grace window to resolve before failing them; requests that resolve
inside the window succeed, and the teardown still completes promptly when
the connector never answers.
"""

import asyncio

import pytest

from gateway.relay.ws_transport import WebSocketRelayTransport


def _transport():
    t = object.__new__(WebSocketRelayTransport)
    t._closing = False
    t._supervisor = None
    t._reader = None
    t._ws = None
    t._pending = {}
    t._going_idle_ack = None
    return t


@pytest.mark.asyncio
async def test_disconnect_waits_for_inflight_outbound():
    """An outbound frame whose result arrives during teardown must resolve
    successfully, not be failed with 'relay transport closed'."""
    t = _transport()
    fut = asyncio.get_running_loop().create_future()
    t._pending["req-1"] = fut

    async def _connector_answers_soon():
        await asyncio.sleep(0.05)
        if not fut.done():
            fut.set_result({"success": True, "message_id": "m9"})

    answer_task = asyncio.create_task(_connector_answers_soon())
    await t.disconnect()
    await answer_task

    assert fut.done()
    assert fut.exception() is None, (
        "in-flight finalize edit was failed at teardown instead of being "
        "allowed to complete within the drain grace window"
    )
    assert fut.result()["success"] is True


@pytest.mark.asyncio
async def test_disconnect_still_bounded_when_connector_silent():
    """No answer ever arrives: the drain must give up within the grace
    bound and fail the future so callers don't hang."""
    t = _transport()
    fut = asyncio.get_running_loop().create_future()
    t._pending["req-2"] = fut

    started = asyncio.get_running_loop().time()
    await t.disconnect()
    elapsed = asyncio.get_running_loop().time() - started

    assert fut.done() and fut.exception() is not None
    assert elapsed < 10.0, f"teardown took {elapsed:.1f}s — grace must be bounded"


def test_drain_grace_clamps_to_disconnect_budget(monkeypatch):
    """Drain + the three sequential teardown awaits must fit the runner's
    adapter-disconnect budget (default 5s): grace = budget - 3*1.0s - margin."""
    from gateway.relay import ws_transport as wt

    monkeypatch.delenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", raising=False)
    grace = wt._disconnect_drain_grace_s()
    budget = 5.0
    reserved = 3 * wt._TEARDOWN_AWAIT_TIMEOUT_S
    assert grace + reserved < budget, (
        f"grace {grace}s + teardown awaits {reserved}s exceeds the runner's "
        f"{budget}s disconnect budget — wait_for would cancel teardown "
        "mid-drain and skip the fail-pending loop"
    )
    assert grace > 0, "budget default must still leave usable drain time"


def test_drain_grace_honors_env_budget(monkeypatch):
    from gateway.relay import ws_transport as wt

    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "12")
    assert wt._disconnect_drain_grace_s() == wt._DISCONNECT_DRAIN_GRACE_S
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "3.5")
    grace = wt._disconnect_drain_grace_s()
    assert 0 <= grace <= 3.5 - 3 * wt._TEARDOWN_AWAIT_TIMEOUT_S
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "1")
    assert wt._disconnect_drain_grace_s() == 0.0
