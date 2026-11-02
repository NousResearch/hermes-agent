"""Only a local pre-write rejection can refund a replay attempt."""

import json
from types import SimpleNamespace

import pytest

from gateway import delivery_ledger as ledger
from gateway.config import Platform
from gateway.relay.ws_transport import WebSocketRelayTransport
from tests.gateway.test_relay_ledger_recovery import record, setup_runner


def wire_transport(connector, response):
    transport = WebSocketRelayTransport.__new__(WebSocketRelayTransport)
    transport._closing = False
    transport._ws = None
    transport._pending = {}
    transport._identities = connector._identities
    transport._outbound_timeout_s = 10
    frames = []

    async def send(frame):
        body = json.loads(frame)
        frames.append(body)
        transport._pending[body["requestId"]].set_result(dict(response))

    async def outbound(action, *, platform=None):
        return await transport._request_response(action, platform=platform)

    connector.send_outbound = outbound
    return transport, SimpleNamespace(send=send), frames


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", [False, True])
async def test_repeated_local_rejections_preserve_budget_until_a_real_send(
    tmp_path, monkeypatch, closed
):
    runner, _, connector = setup_runner(tmp_path, monkeypatch)
    transport, socket, frames = wire_transport(
        connector, {"success": True, "message_id": "m1"}
    )
    transport._closing = closed
    record()
    ledger.mark_failed("owed", "relay transport not connected")
    for _ in range(ledger.MAX_ATTEMPTS + 2):
        assert (
            await runner._redeliver_failed_obligations_for_platform(Platform.RELAY) == 0
        )
        with ledger._transaction() as db:
            assert db.execute(
                "SELECT state, attempts FROM delivery_obligations"
            ).fetchone() == ("failed", 0)
        assert ledger.pending_reconnects() == {("slack", "default")}
    assert not frames
    transport._closing, transport._ws = False, socket
    assert await runner._redeliver_failed_obligations_for_platform(Platform.RELAY) == 1
    assert len(frames) == 1
    with ledger._transaction() as db:
        assert db.execute(
            "SELECT state, attempts FROM delivery_obligations"
        ).fetchone() == ("delivered", 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"success": False, "error": "relay transport not connected", "not_sent": True},
        {
            "success": False,
            "error": "relay transport connection lost",
            "ambiguous": True,
        },
        {"success": False, "error": "relay send failed: ConnectionClosed"},
        {
            "success": False,
            "error": "partial send",
            "message_id": "partial",
            "not_sent": True,
        },
    ],
)
async def test_remote_rejection_or_partial_delivery_cannot_claim_a_local_refund(
    tmp_path, monkeypatch, response
):
    runner, _, connector = setup_runner(tmp_path, monkeypatch)
    transport, socket, frames = wire_transport(connector, response)
    transport._ws = socket
    record()
    ledger.mark_failed("owed", "send_path_degraded")
    assert await runner._redeliver_failed_obligations_for_platform(Platform.RELAY) == 0
    assert len(frames) == 1
    with ledger._transaction() as db:
        assert db.execute(
            "SELECT state, attempts FROM delivery_obligations"
        ).fetchone() == ("failed", 1)


@pytest.mark.asyncio
async def test_recovery_does_not_seal_another_turn_streaming_in_the_same_thread(
    tmp_path, monkeypatch
):
    runner, adapter, connector = setup_runner(tmp_path, monkeypatch)
    key = adapter._draft_key("chat", {"thread_id": "thread"})
    adapter._open_draft_by_chat[key] = "live-turn"
    record()
    ledger.mark_failed("owed", "send_path_degraded")
    assert await runner._redeliver_failed_obligations_for_platform(Platform.RELAY) == 1
    assert [frame["op"] for frame in connector.sent] == ["send"]
    assert adapter._open_draft_by_chat[key] == "live-turn"
    assert "_interim_send" not in connector.sent[0]["metadata"]
