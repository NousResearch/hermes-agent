"""A reconnect must not outrun the durable write making its reply claimable."""

import asyncio
import threading
from unittest.mock import Mock

import pytest

from gateway import delivery_ledger as ledger
from gateway.run_delivery_recovery import schedule_relay_recovery
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.relay.ws_transport import WebSocketRelayTransport
from tests.gateway.relay.test_relay_adapter import make_desc
from tests.gateway.test_relay_ledger_recovery import record, setup_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("ready", [False, True])
async def test_reconnect_before_claim_refund_still_delivers(
    tmp_path, monkeypatch, ready
):
    runner, adapter, connector = setup_runner(tmp_path, monkeypatch)
    connector.delivery_ready = ready
    record()
    ledger.mark_failed("owed", "send_path_degraded")
    claimed = ledger.sweep_failed_for_runtime("slack")
    assert len(claimed) == 1
    live = runner.adapters
    runner.adapters = {}
    entered, release = asyncio.Event(), asyncio.Event()
    original = runner._release_runtime_claim_quiet

    async def held_refund(*args, **kwargs):
        entered.set()
        await release.wait()
        await original(*args, **kwargs)

    monkeypatch.setattr(runner, "_release_runtime_claim_quiet", held_refund)
    replay = asyncio.create_task(runner._redeliver_claimed_obligations(claimed))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        runner.adapters = live
        schedule_relay_recovery(adapter)
        await asyncio.gather(*list(runner._background_tasks))
        assert not connector.sent  # the durable row is still attempting
        release.set()
        await asyncio.wait_for(replay, 10)
        await asyncio.gather(*list(runner._background_tasks))
        with ledger._connect() as db:
            row = db.execute(
                "SELECT state, attempts FROM delivery_obligations"
            ).fetchone()
        assert row == (("delivered", 1) if ready else ("failed", 0))
        assert len(connector.sent) == int(ready)
    finally:
        release.set()
        await replay


@pytest.mark.asyncio
async def test_relay_disconnect_waits_for_reconnect_without_burning_retry_budget(
    tmp_path, monkeypatch
):
    runner, adapter, connector = setup_runner(tmp_path, monkeypatch)
    connector.delivery_ready = False
    runner._schedule_flood_redelivery = Mock()
    record()
    event = MessageEvent(
        text="question",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id="chat"),
    )
    await adapter._finalize_delivery_obligation(
        "owed",
        SendResult(success=False, error="relay transport not connected"),
        event,
        adapter,
    )
    assert ledger.pending_retries() == []
    runner._schedule_flood_redelivery.assert_not_called()
    connector.delivery_ready = True
    schedule_relay_recovery(adapter)
    await asyncio.gather(*list(runner._background_tasks))
    assert len(connector.sent) == 1
    with ledger._connect() as db:
        assert db.execute(
            "SELECT state, attempts FROM delivery_obligations"
        ).fetchone() == ("delivered", 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["producer", "replay"])
@pytest.mark.parametrize("owner", ["default", "secondary"])
async def test_same_relay_reconnect_before_failure_write_still_delivers(
    tmp_path, monkeypatch, origin, owner
):
    runner, adapter, connector = setup_runner(tmp_path, monkeypatch)
    adapter._owner_profile = owner
    if owner != "default":
        runner._profile_adapters = {owner: runner.adapters}
        runner._profile_configs = {owner: runner.config}
        runner.adapters = {}
    runner._schedule_flood_redelivery = Mock()
    connector.delivery_ready = False
    record(profile=owner)
    error = "relay transport connection lost"
    if origin == "replay":
        ledger.mark_failed("owed", "send_path_degraded")
        claimed = ledger.sweep_failed_for_runtime("slack", profile=owner)
        connector.next_send_result = {"success": False, "error": error}
    else:
        ledger.mark_attempting("owed")
    entered, release = threading.Event(), threading.Event()
    original = ledger.mark_failed

    def held_failure(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(ledger, "mark_failed", held_failure)
    event = MessageEvent(
        text="question",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id="chat"),
    )
    finalizer = asyncio.create_task(
        runner._redeliver_claimed_obligations(claimed)
        if origin == "replay"
        else adapter._finalize_delivery_obligation(
            "owed", SendResult(success=False, error=error), event, adapter
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        connector.delivery_ready = True
        connector.next_send_result = {"success": True, "message_id": "recovered"}
        schedule_relay_recovery(adapter)
        await asyncio.gather(*list(runner._background_tasks))
        assert len(connector.sent) == int(origin == "replay")
        release.set()
        await asyncio.wait_for(finalizer, 10)
        await asyncio.gather(*list(runner._background_tasks))
        assert len(connector.sent) == 1 + int(origin == "replay")
        with ledger._connect() as db:
            assert db.execute(
                "SELECT state, attempts FROM delivery_obligations"
            ).fetchone() == ("delivered", 1 + int(origin == "replay"))
    finally:
        release.set()
        await finalizer


@pytest.mark.parametrize(
    "unready", [None, "_ws", "_descriptor", "_closing", "_dormant", "_auth_revoked"]
)
def test_relay_delivery_readiness_requires_a_live_authenticated_socket(unready):
    transport = WebSocketRelayTransport.__new__(WebSocketRelayTransport)
    transport._ws = object()
    transport._descriptor = make_desc(platform="slack")
    transport._closing = transport._dormant = transport._auth_revoked = False
    if unready:
        setattr(transport, unready, None if unready in {"_ws", "_descriptor"} else True)
    assert transport.delivery_ready is (unready is None)
