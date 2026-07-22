"""Relay passthrough-over-WS forwarding (Phase 5 §5.1).

Proves the gateway side of §5.1: a connector-forwarded passthrough request
(Discord interaction, Twilio, …) arrives over the SAME outbound /relay WS as
inbound messages (a hosted gateway has no public inbound port), and the relay
adapter handles it — decoding the byte-preserved body and routing a Discord
interaction through the normal agent path (handle_message).

Mirrors test_relay_interrupt.py's wiring discipline (connect() registers the
connector->gateway handlers on the transport).
"""

from __future__ import annotations

import base64
import json

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.ws_transport import PassthroughForward, _passthrough_from_wire

from tests.gateway.relay.stub_connector import StubConnector


def _desc() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label="Discord",
        max_message_length=2000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="discord",
        len_unit="chars",
    )


@pytest.fixture
def adapter():
    return RelayAdapter(PlatformConfig(), _desc(), transport=StubConnector(_desc()))


def _interaction_forward(payload: dict, *, profile: str | None = None) -> PassthroughForward:
    body = json.dumps(payload).encode("utf-8")
    return PassthroughForward(
        platform="discord",
        bot_id="appShared",
        method="POST",
        path="/interactions/discord/appShared",
        headers=[("content-type", "application/json")],
        body=body,
        profile=profile,
    )


def test_passthrough_from_wire_byte_preserves_body():
    """The wire frame's base64 body decodes back to the exact bytes (parity with
    the connector's toPassthroughForward)."""
    original = json.dumps({"type": 2, "data": {"name": "ping"}, "guild_id": "g1"}).encode("utf-8")
    wire = {
        "platform": "discord",
        "botId": "appShared",
        "method": "POST",
        "path": "/interactions/discord/appShared",
        "headers": [["content-type", "application/json"]],
        "bodyB64": base64.b64encode(original).decode("ascii"),
    }
    fwd = _passthrough_from_wire(wire)
    assert fwd.platform == "discord"
    assert fwd.bot_id == "appShared"
    assert fwd.body == original
    assert fwd.headers == [("content-type", "application/json")]


def test_passthrough_from_wire_tolerates_malformed_body():
    """A non-base64 body must not raise (the reader must never crash)."""
    fwd = _passthrough_from_wire({"platform": "x", "bodyB64": "!!!not base64!!!"})
    assert fwd.body == b""


def test_passthrough_from_wire_stamps_routed_profile():
    """A connector-routed profile on the wire frame lands on PassthroughForward.

    Mirrors _event_from_wire's profile stamping for the ``inbound`` frame
    (#60586) — the passthrough plane needs the same carry-through so a
    Team-Gateway's Discord interactions route to the same profile a plain
    message would.
    """
    wire = {
        "platform": "discord",
        "botId": "appShared",
        "method": "POST",
        "path": "/interactions/discord/appShared",
        "headers": [],
        "bodyB64": "",
        "profile": "reviewer",
    }
    fwd = _passthrough_from_wire(wire)
    assert fwd.profile == "reviewer"


def test_passthrough_from_wire_profile_absent_is_none():
    """No ``profile`` on the wire (single-profile gateway) → None.

    Back-compat: a connector that never sets ``profile`` yields the legacy
    behaviour, and session keys stay in the ``agent:main`` namespace.
    """
    fwd = _passthrough_from_wire(
        {"platform": "discord", "botId": "appShared", "method": "POST", "path": "/x", "headers": [], "bodyB64": ""}
    )
    assert fwd.profile is None


@pytest.mark.asyncio
async def test_connect_wires_passthrough_handler_over_ws(adapter):
    """connect() registers the passthrough handler on the transport so a
    connector-delivered passthrough_forward frame reaches the adapter."""
    await adapter.connect()
    stub = adapter._transport
    assert stub._passthrough is not None


@pytest.mark.asyncio
async def test_discord_interaction_routes_through_handle_message(adapter, monkeypatch):
    """A forwarded Discord application-command interaction is decoded and routed
    through the normal agent path (handle_message) with a correct session source."""
    await adapter.connect()
    stub = adapter._transport

    seen = []

    async def fake_handle(event):
        seen.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    fwd = _interaction_forward(
        {
            "id": "interaction-1",
            "type": 2,  # APPLICATION_COMMAND
            "channel_id": "chan-9",
            "guild_id": "guild-7",
            "data": {"name": "summarize"},
            "member": {"user": {"id": "user-3", "username": "ben"}},
        }
    )
    await stub.push_passthrough(fwd, buffer_id=None)

    assert len(seen) == 1
    ev = seen[0]
    assert ev.text == "summarize"
    assert ev.source.chat_id == "chan-9"
    assert ev.source.scope_id == "guild-7"
    assert ev.source.user_id == "user-3"
    assert ev.source.chat_type == "channel"
    # Scope captured so the agent's reply re-asserts scope_id for egress.
    assert adapter._scope_by_chat.get("chan-9") == "guild-7"
    # Back-compat: no profile on the forward (single-profile gateway) → None,
    # session keys stay in the legacy agent:main namespace.
    assert ev.source.profile is None


@pytest.mark.asyncio
async def test_discord_interaction_stamps_routed_profile(adapter, monkeypatch):
    """A connector-routed profile on the passthrough forward lands on the
    resulting event's SessionSource, the same way it does for a plain relayed
    message (#60586) — so a Team-Gateway's Discord slash-command/button/modal
    routes to the same profile a plain message would, instead of always
    falling back to agent:main."""
    await adapter.connect()
    stub = adapter._transport

    seen = []

    async def fake_handle(event):
        seen.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    fwd = _interaction_forward(
        {
            "id": "interaction-2",
            "type": 2,  # APPLICATION_COMMAND
            "channel_id": "chan-9",
            "guild_id": "guild-7",
            "data": {"name": "summarize"},
            "member": {"user": {"id": "user-3", "username": "ben"}},
        },
        profile="reviewer",
    )
    await stub.push_passthrough(fwd, buffer_id=None)

    assert len(seen) == 1
    assert seen[0].source.profile == "reviewer"


@pytest.mark.asyncio
async def test_message_component_interaction_uses_custom_id(adapter, monkeypatch):
    """A MESSAGE_COMPONENT (button) interaction surfaces its custom_id as text."""
    await adapter.connect()
    stub = adapter._transport
    seen = []

    async def fake_handle(event):
        seen.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    fwd = _interaction_forward(
        {
            "id": "i2",
            "type": 3,  # MESSAGE_COMPONENT
            "channel_id": "c2",
            "guild_id": "g2",
            "data": {"custom_id": "approve_btn"},
            "member": {"user": {"id": "u2", "username": "x"}},
        }
    )
    await stub.push_passthrough(fwd)
    assert len(seen) == 1
    assert seen[0].text == "approve_btn"


@pytest.mark.asyncio
async def test_malformed_interaction_body_does_not_raise(adapter, monkeypatch):
    """A non-JSON forward is logged and dropped — never crashes the read loop."""
    await adapter.connect()
    stub = adapter._transport
    called = []

    async def fake_handle(event):
        called.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    bad = PassthroughForward(
        platform="discord",
        bot_id="appShared",
        method="POST",
        path="/x",
        headers=[],
        body=b"not json",
    )
    await stub.push_passthrough(bad)  # must not raise
    assert called == []


@pytest.mark.asyncio
async def test_non_discord_forward_dropped_cleanly(adapter, monkeypatch):
    """A platform with no gateway-side handler yet (e.g. twilio) is dropped, not raised."""
    await adapter.connect()
    stub = adapter._transport
    called = []

    async def fake_handle(event):
        called.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    fwd = PassthroughForward(
        platform="twilio",
        bot_id="bot1",
        method="POST",
        path="/webhooks/twilio/seg",
        headers=[],
        body=b"From=+1&Body=hi",
    )
    await stub.push_passthrough(fwd)  # must not raise
    assert called == []
