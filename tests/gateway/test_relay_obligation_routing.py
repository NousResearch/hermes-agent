"""Replay uses the failed send's identity across restarts and shared sessions."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from gateway.relay.transport import RelayTransport
from gateway.relay.ws_transport import WebSocketRelayTransport
from gateway.session import SessionSource
from tests.gateway.relay.stub_connector import StubConnector
from tests.gateway.relay.test_relay_adapter import make_desc
from tests.gateway.test_delivery_ledger import (
    TestGatewayRedeliverySweep as _ReplayHarness,
    _orphan,
    _record,
    _row,
)


ROUTING = {"scope_id": "workspace", "user_id": "second-user", "profile": "worker"}


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(dl, "ledger_enabled", lambda: True)


class RoutedConnector(StubConnector):
    expected_routing = ROUTING

    async def send_outbound(self, action, *, platform=None):
        metadata = action.get("metadata", {})
        actual = {key: metadata[key] for key in ROUTING if key in metadata}
        if actual != self.expected_routing:
            return {
                "success": False,
                "error": "target not routed to an onboarded tenant",
            }
        return await super().send_outbound(action, platform=platform)


def relay(connector_type=RoutedConnector):
    descriptor = make_desc(platform="slack")
    connector = connector_type(descriptor)
    adapter = RelayAdapter(
        PlatformConfig(), descriptor, cast(RelayTransport, connector)
    )
    return adapter, connector


@pytest.mark.asyncio
@pytest.mark.parametrize("origin_kind", ["missing", "mismatched", "old-author"])
async def test_failed_send_snapshot_survives_cold_cache_and_session_origin(origin_kind):
    original, transport = relay()
    # The adapter cache already describes another user. This turn's source wins.
    original._dm_user_by_chat["C1"] = "first-user"
    original._scope_by_chat["C1"] = "workspace"
    original._platform_by_chat["C1"] = "slack"
    transport.next_send_result = {"success": False, "error": "send_path_degraded"}
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        thread_id="171.001",
        scope_id="workspace",
        user_id="second-user",
        profile="worker",
    )
    event = MessageEvent(
        text="question",
        message_type=MessageType.TEXT,
        source=source,
        message_id="turn-2",
    )
    result, _ = await original.send_final_ledgered(
        event,
        "shared-session",
        "answer",
        {"thread_id": "171.001"},
        reply_to=None,
    )
    assert not result.success
    with dl._connect() as conn:
        oid, raw = conn.execute(
            "SELECT obligation_id, routing_metadata FROM delivery_obligations"
        ).fetchone()
    assert json.loads(raw) == ROUTING
    assert {key: transport.sent[0]["metadata"][key] for key in ROUTING} == ROUTING
    _orphan(oid)

    replacement, connector = relay()
    runner = _ReplayHarness._runner()
    runner.adapters = {Platform.RELAY: replacement}
    origin = (
        None
        if origin_kind == "missing"
        else SessionSource(
            platform=Platform.SLACK,
            chat_id="other-chat" if origin_kind == "mismatched" else "C1",
            thread_id="171.001",
            user_id="first-user",
            scope_id="old-workspace",
            profile="old-profile",
        )
    )
    runner._async_session_store.lookup_by_session_key = AsyncMock(
        return_value=SimpleNamespace(origin=origin)
    )
    assert await runner._redeliver_pending_obligations() == 1
    assert _row(oid)["state"] == "delivered"
    assert len(connector.sent) == 1
    assert {key: connector.sent[0]["metadata"][key] for key in ROUTING} == ROUTING


@pytest.mark.asyncio
@pytest.mark.parametrize("routing", [None, {}, {"profile": "worker"}])
async def test_unrouted_legacy_rows_defer_without_spending_attempts(routing):
    _record(routing_metadata=routing)
    _orphan("ob-1")
    adapter, connector = relay()
    runner = _ReplayHarness._runner()
    runner._async_session_store.lookup_by_session_key = AsyncMock(return_value=None)
    runner.adapters = {Platform.RELAY: adapter}
    assert await runner._redeliver_pending_obligations() == 0
    for _ in range(dl.MAX_ATTEMPTS + 1):
        assert (
            await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 0
        )
    assert _row("ob-1")["attempts"] == 0
    assert _row("ob-1")["state"] == "failed"
    assert _row("ob-1")["last_error"] == "relay routing metadata unavailable"
    assert connector.sent == []
    assert dl.pending_retries() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "routing", [ROUTING, {"scope_id": "workspace"}, {"user_id": "second-user"}]
)
async def test_replay_does_not_borrow_live_cache_identity(routing):
    _record(routing_metadata=routing)
    _orphan("ob-1")
    adapter, connector = relay()
    connector.expected_routing = routing
    adapter._scope_by_chat["C1"] = "new-workspace"
    adapter._dm_user_by_chat["C1"] = "new-user"
    adapter._profile_by_chat["C1"] = "new-profile"
    adapter._open_draft_by_chat["C1:new-turn"] = 123
    runner = _ReplayHarness._runner()
    runner._async_session_store.lookup_by_session_key = AsyncMock(return_value=None)
    runner.adapters = {Platform.RELAY: adapter}
    assert await runner._redeliver_pending_obligations() == 1
    assert _row("ob-1")["state"] == "delivered"
    assert connector.sent[0]["op"] == "send"
    assert adapter._open_draft_by_chat == {"C1:new-turn": 123}


def test_ledger_snapshot_excludes_unrelated_metadata():
    _record(
        routing_metadata={
            **ROUTING,
            "authorization": "synthetic-secret",
            "thread_id": "171.001",
        }
    )
    _orphan("ob-1")
    assert dl.sweep_recoverable()[0]["routing_metadata"] == ROUTING


def test_existing_ledger_migrates_without_inventing_routing():
    _record()
    _orphan("ob-1")
    with dl._connect() as conn:
        conn.execute("ALTER TABLE delivery_obligations DROP COLUMN routing_metadata")
    row = dl.sweep_recoverable()[0]
    assert row["content"] == "the final answer"
    assert row["routing_metadata"] == {}


def replay_runner(adapter):
    runner = _ReplayHarness._runner()
    runner._async_session_store.lookup_by_session_key = AsyncMock(return_value=None)
    runner.adapters = {Platform.RELAY: adapter}
    runner._running = True
    runner._background_tasks = set()
    return runner


class FakeSocket:
    """Stands in for the WebSocket so the real transport writes frames and counts them."""

    def __init__(self, transport, reply):
        self.transport, self.reply, self.frames = transport, reply, []

    async def send(self, data):
        frame = json.loads(data)
        self.frames.append(frame)
        if self.reply == "raise":
            raise ConnectionResetError("socket closed during write")
        if isinstance(self.reply, dict):
            self.transport._pending[frame["requestId"]].set_result(dict(self.reply))
        # None: written, never acknowledged


def ws_relay(monkeypatch, *, socket="absent", closing=False):
    """A RelayAdapter over the real WebSocket transport. ``socket`` is ``"absent"`` or a reply."""
    transport = WebSocketRelayTransport(
        "ws://relay.invalid", "slack", "bot", reconnect=False, outbound_timeout_s=0.05
    )
    adapter = RelayAdapter(PlatformConfig(), make_desc(platform="slack"), transport)
    # The handshake had acknowledged the platform when the claim was resolved.
    monkeypatch.setattr(transport, "is_platform_ready", lambda platform: True)
    transport._closing = closing
    if socket != "absent":
        transport._ws = FakeSocket(transport, socket)
    return adapter, transport


async def boot_replay(adapter):
    _record(routing_metadata=ROUTING)
    _orphan("ob-1")
    runner = replay_runner(adapter)
    await runner._redeliver_pending_obligations()
    return runner, _row("ob-1")


REFUSAL_TEXT = {"success": False, "error": "relay transport not connected"}


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", [False, True])
async def test_a_send_the_local_transport_refused_does_not_spend_an_attempt(monkeypatch, closing):
    adapter, _ = ws_relay(monkeypatch, closing=closing)

    _, refused = await boot_replay(adapter)

    assert (refused["state"], refused["attempts"]) == ("failed", 0)
    assert refused["last_error"] in {"relay transport not connected", "relay transport closed"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "socket",
    [
        "raise",  # the write raised: bytes may be out
        None,  # written, acknowledgement lost
        REFUSAL_TEXT,  # the connector answers with the local refusal's own words
        {**REFUSAL_TEXT, "not_sent": True},  # ... and claims it was not sent
        {"success": False, "error": "relay transport connection lost"},
        {"success": False, "error": "partial", "message_id": "m-1"},
    ],
)
async def test_a_reply_that_may_have_reached_the_platform_keeps_its_attempt(monkeypatch, socket):
    adapter, transport = ws_relay(monkeypatch, socket=socket)

    _, row = await boot_replay(adapter)

    assert transport._ws.frames, "the frame must have been written for this case to mean anything"
    assert (row["state"], row["attempts"]) == ("failed", 1)


@pytest.mark.asyncio
async def test_an_offline_relay_cannot_abandon_a_reply_it_never_sent(monkeypatch):
    adapter, transport = ws_relay(monkeypatch)
    runner, _ = await boot_replay(adapter)

    # More local refusals than the whole budget: the reply must still be owed, not abandoned.
    for _ in range(dl.MAX_ATTEMPTS + 1):
        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK, relay_adapter=adapter) == 0
    owed = _row("ob-1")
    assert (owed["state"], owed["attempts"]) == ("failed", 0)

    transport._ws = FakeSocket(transport, {"success": True, "message_id": "m1"})
    assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK, relay_adapter=adapter) == 1
    assert _row("ob-1")["state"] == "delivered"


@pytest.mark.asyncio
async def test_a_failed_delivery_record_is_retried_without_sending_the_reply_again(monkeypatch):
    _record(routing_metadata=ROUTING)
    _orphan("ob-1")
    adapter, connector = relay()
    runner = replay_runner(adapter)
    record, writes = dl.mark_delivered, []

    def busy_once(obligation_id):
        writes.append(obligation_id)
        if len(writes) == 1:
            raise sqlite3.OperationalError("database is locked")
        return record(obligation_id)

    monkeypatch.setattr(dl, "mark_delivered", busy_once)
    monkeypatch.setattr("gateway.relay.obligation_replay.SETTLE_RETRY_DELAYS", (0.001, 0.001), raising=False)

    assert await runner._redeliver_pending_obligations() == 1

    assert _row("ob-1")["state"] == "delivered"
    assert len(writes) == 2
    assert [frame["op"] for frame in connector.sent] == ["send"]


@pytest.mark.asyncio
async def test_handshakes_during_a_running_replay_share_it_and_get_one_more_pass():
    from gateway.relay.obligation_replay import _schedule_replay

    adapter, _ = relay()
    runner = replay_runner(adapter)
    passes, release = [], asyncio.Event()

    async def sweep(platform, *, relay_adapter=None):
        passes.append(platform)
        await release.wait()
        return 0

    runner._redeliver_failed_obligations_for_platform = sweep
    for _ in range(4):
        _schedule_replay(runner, adapter, Platform.SLACK)
    await asyncio.sleep(0)
    assert len(passes) == 1 and len(runner._background_tasks) == 1

    release.set()
    await asyncio.gather(*runner._background_tasks)

    assert passes == [Platform.SLACK, Platform.SLACK]
    # Nothing is left behind, so a later handshake starts a fresh worker.
    _schedule_replay(runner, adapter, Platform.SLACK)
    await asyncio.gather(*runner._background_tasks)
    assert len(passes) == 3


@pytest.mark.asyncio
async def test_a_relay_fronted_plugin_platform_is_replayed_at_boot_and_on_replacement():
    from gateway.relay.obligation_replay import replay_on_relay_replacement

    plugin = Platform("irc")
    assert plugin not in list(Platform)  # a plugin platform is not an enum iteration member
    _record(platform="irc", routing_metadata=ROUTING)
    _orphan("ob-1")
    adapter, connector = relay()
    connector._identities = [("irc", "")]
    runner = replay_runner(adapter)

    assert await runner._redeliver_pending_obligations() == 1
    assert _row("ob-1")["state"] == "delivered"
    assert connector.sent[0]["op"] == "send"

    passes = []

    async def sweep(platform, *, relay_adapter=None):
        passes.append(platform)
        return 0

    runner._redeliver_failed_obligations_for_platform = sweep
    replay_on_relay_replacement(runner, adapter)
    await asyncio.gather(*runner._background_tasks)
    assert passes == [plugin]
