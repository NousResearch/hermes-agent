"""Replay uses the failed send's identity across restarts and shared sessions."""

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from gateway.relay.transport import RelayTransport
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
