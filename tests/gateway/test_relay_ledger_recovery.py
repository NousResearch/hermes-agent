"""Real ledger recovery through a relay with no inbound routing cache."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as ledger
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.transport import RelayTransport
from gateway.relay.ws_transport import WebSocketRelayTransport
from gateway.run import GatewayRunner
from tests.gateway.relay.stub_connector import StubConnector
from tests.gateway.relay.test_relay_adapter import make_desc


def setup_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.db")
    monkeypatch.setattr(ledger, "_owner_stamp", lambda: (424242, 1234.5))
    connector = StubConnector(make_desc(platform="slack"))
    connector._identities = [("slack", "bot-slack"), ("discord", "bot-discord")]
    adapter = RelayAdapter(
        PlatformConfig(enabled=True),
        make_desc(platform="slack"),
        cast(RelayTransport, connector),
    )
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.RELAY: adapter.config})
    runner.adapters = {Platform.RELAY: adapter}
    runner._profile_adapters = {}
    runner._primary_profile_name = "default"
    runner._running_agents = {}
    runner._running = True
    runner.session_store = cast(Any, None)
    runner._async_session_store = cast(
        Any, SimpleNamespace(_store=None, clear_resume_pending=AsyncMock())
    )
    runner._arm_flood_timers_for_waiting_rows = AsyncMock()
    adapter.gateway_runner = runner
    return runner, adapter, connector


def record(profile="default", logical="slack", namespace="main"):
    ledger.record_obligation(
        obligation_id="owed",
        session_key=f"agent:{namespace}:{logical}:channel:chat",
        platform=logical,
        chat_id="chat",
        thread_id="thread",
        content="stored answer",
        adapter_profile=profile,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("logical", ["slack", "discord"])
async def test_reconnect_replays_logical_platform_once(tmp_path, monkeypatch, logical):
    runner, _, connector = setup_runner(tmp_path, monkeypatch)
    record(logical=logical, namespace="satellite")
    ledger.mark_failed("owed", "send_path_degraded")
    assert await runner._redeliver_failed_obligations_for_platform(Platform.RELAY) == 1
    assert await runner._redeliver_failed_obligations_for_platform(Platform.RELAY) == 0
    assert connector.sent_platforms == [logical]
    assert connector.sent[0]["metadata"]["thread_id"] == "thread"
    assert connector.sent[0]["metadata"]["profile"] == "satellite"
    assert connector.sent[0]["content"].endswith("stored answer")
    runner._async_session_store.clear_resume_pending.assert_awaited_once()


@pytest.mark.asyncio
async def test_boot_claims_relay_fronted_row_without_spending_other_bots_budget(
    tmp_path, monkeypatch
):
    runner, _, connector = setup_runner(tmp_path, monkeypatch)
    record()
    with ledger._connect() as db:
        db.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1"
        )
    assert await runner._redeliver_pending_obligations() == 1
    assert connector.sent_platforms == ["slack"]
    record(profile="offline")
    with ledger._connect() as db:
        db.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1"
        )
    assert await runner._redeliver_pending_obligations() == 0
    with ledger._connect() as db:
        assert (
            db.execute("SELECT attempts FROM delivery_obligations").fetchone()[0] == 0
        )


@pytest.mark.asyncio
async def test_replay_does_not_use_unadvertised_platform(tmp_path, monkeypatch):
    runner, _, connector = setup_runner(tmp_path, monkeypatch)
    record(logical="telegram")
    ledger.mark_failed("owed", "send_path_degraded")
    assert await runner._redeliver_failed_obligations_for_platform(Platform.RELAY) == 0
    assert not connector.sent


@pytest.mark.asyncio
async def test_replay_uses_secondary_credentials_and_preserves_runtime_profile(
    tmp_path, monkeypatch
):
    runner, _, primary = setup_runner(tmp_path, monkeypatch)
    secondary = StubConnector(make_desc(platform="slack"))
    adapter = RelayAdapter(
        PlatformConfig(enabled=True),
        make_desc(platform="slack"),
        cast(RelayTransport, secondary),
    )
    runner._profile_adapters = {"secondary": {Platform.RELAY: adapter}}
    runner._profile_configs = {
        "secondary": GatewayConfig(platforms={Platform.RELAY: adapter.config})
    }
    record(profile="secondary", namespace="work")
    ledger.mark_failed("owed", "send_path_degraded")
    assert await runner._redeliver_failed_obligations_for_platform(Platform.RELAY) == 0
    assert (
        await runner._redeliver_failed_obligations_for_platform(
            Platform.RELAY, profile="secondary"
        )
        == 1
    )
    assert not primary.sent
    assert secondary.sent_platforms == ["slack"]
    assert secondary.sent[0]["metadata"]["profile"] == "work"


@pytest.mark.asyncio
async def test_replay_prefers_existing_native_adapter(tmp_path, monkeypatch):
    runner, _, connector = setup_runner(tmp_path, monkeypatch)
    native = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(success=True)))
    runner.adapters[Platform.SLACK] = native
    record()
    ledger.mark_failed("owed", "send_path_degraded")
    assert await runner._redeliver_failed_obligations_for_platform(Platform.RELAY) == 1
    assert not connector.sent
    assert native.send.await_args.kwargs["chat_id"] == "chat"
    assert native.send.await_args.kwargs["metadata"] == {"thread_id": "thread"}


@pytest.mark.asyncio
async def test_transport_reconnect_notifies_after_handshake(tmp_path, monkeypatch):
    runner, adapter, connector = setup_runner(tmp_path, monkeypatch)
    transport = WebSocketRelayTransport.__new__(WebSocketRelayTransport)
    transport._closing = False
    transport._dormant = False
    transport._reconnect_backoff_s = 0
    transport._await_redial_hold = AsyncMock()
    transport._dial_and_start = AsyncMock()
    transport._dial_generation = 2
    transport._descriptor = None
    transport._descriptor_ready = asyncio.get_running_loop().create_future()
    transport._descriptors_by_platform = {}
    connector.set_reconnect_handler = transport.set_reconnect_handler
    assert await adapter.connect()
    record()
    ledger.mark_failed("owed", "send_path_degraded")
    await transport._reconnect_loop()
    assert not getattr(runner, "_background_tasks", set())
    frame = {"descriptor": json.loads(make_desc(platform="slack").to_json())}
    await transport._on_descriptor(frame)
    await transport._on_descriptor(frame)
    pending = list(runner._background_tasks)
    assert len(pending) == 1
    await asyncio.gather(*pending)
    assert transport._descriptor_ready.done()
    assert connector.sent_platforms == ["slack"]
