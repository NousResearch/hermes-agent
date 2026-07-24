"""Per-platform capability descriptors on the relay (multi-platform Phase 1.5).

The bug class: one relay adapter fronts N platforms on one WS, but the
capability surface (``MAX_MESSAGE_LENGTH`` / ``message_len_fn``) was a SCALAR
from whichever descriptor resolved the handshake — so a Discord chat on a
gateway whose primary identity was Telegram inherited Telegram's 4,096-char
cap and over-sent into Discord's 2,000-char API 400 (observed live: 2,543 and
2,641-char sends rejected).

Covers:
  - the transport accumulating one descriptor per platform (first = session
    default, later frames must NOT overwrite it),
  - the map resetting on a re-dial,
  - RelayAdapter.max_message_length_for_chat / message_len_fn_for_chat
    resolving from the chat's inbound platform,
  - fallback to the scalar descriptor for unknown chats / transports without
    the map,
  - the stream consumer's _raw_message_limit honoring the per-chat cap.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.session import SessionSource

from tests.gateway.relay.stub_connector import StubConnector


def _descriptor(platform: str, max_len: int, len_unit: str = "chars") -> CapabilityDescriptor:
    return CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform=platform,
        label=platform.title(),
        max_message_length=max_len,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=False,
        markdown_dialect="plain",
        len_unit=len_unit,
    )


DISCORD = _descriptor("discord", 2000)
TELEGRAM = _descriptor("telegram", 4096, len_unit="utf16")


class MultiDescriptorStub(StubConnector):
    """StubConnector extended with the per-platform descriptor map."""

    def __init__(self, primary: CapabilityDescriptor, *others: CapabilityDescriptor) -> None:
        super().__init__(primary)
        self._by_platform = {d.platform: d for d in (primary, *others)}

    def descriptor_for_platform(self, platform: str) -> Optional[CapabilityDescriptor]:
        return self._by_platform.get(platform)


async def _push(stub: StubConnector, platform: Platform, chat_id: str) -> None:
    await stub.push_inbound(
        MessageEvent(
            text="hi",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=platform, chat_id=chat_id, chat_type="dm", user_id="u-1"
            ),
        )
    )


# ───────────────────── transport descriptor accumulation ─────────────────────


def _make_transport():
    from gateway.relay.ws_transport import WebSocketRelayTransport

    return WebSocketRelayTransport(
        "wss://connector.example/relay",
        "telegram",
        "bot-9",
        identities=[("telegram", "bot-9"), ("discord", "app-1")],
    )


@pytest.mark.asyncio
async def test_transport_accumulates_descriptors_first_wins_as_default():
    """One descriptor frame per hello: the map holds each platform's, and the
    scalar `_descriptor` (the handshake result / session default) stays the
    FIRST one — the regression was last-writer-wins across platforms."""
    t = _make_transport()
    loop = asyncio.get_running_loop()
    t._descriptor_ready = loop.create_future()

    frame = {"type": "descriptor", "descriptor": TELEGRAM.__dict__}
    await t._handle_frame(json.dumps(frame))
    frame2 = {"type": "descriptor", "descriptor": DISCORD.__dict__}
    await t._handle_frame(json.dumps(frame2))

    # Per-platform map has both.
    assert t.descriptor_for_platform("telegram").max_message_length == 4096
    assert t.descriptor_for_platform("discord").max_message_length == 2000
    assert t.descriptor_for_platform("slack") is None
    # The session default is the FIRST (primary identity) — NOT overwritten.
    assert t._descriptor.platform == "telegram"
    assert (await t.handshake()).platform == "telegram"


@pytest.mark.asyncio
async def test_transport_descriptor_map_resets_on_redial(monkeypatch):
    """A re-dial starts a fresh handshake generation: stale per-platform
    descriptors must not survive into the new connection."""
    t = _make_transport()
    loop = asyncio.get_running_loop()
    t._descriptor_ready = loop.create_future()
    await t._handle_frame(json.dumps({"type": "descriptor", "descriptor": DISCORD.__dict__}))
    assert t.descriptor_for_platform("discord") is not None

    # Simulate _dial_and_start's reset preamble without a real socket.
    class _FakeWs:
        async def close(self):  # pragma: no cover - not called
            pass

    sent: List[str] = []

    async def _fake_connect(url, **kwargs):
        return _FakeWs()

    async def _fake_send(payload):
        sent.append(payload)

    import gateway.relay.ws_transport as wst

    monkeypatch.setattr(wst, "websockets", type("M", (), {"connect": staticmethod(_fake_connect)}))
    monkeypatch.setattr(t, "_send", _fake_send)
    monkeypatch.setattr(
        t, "_read_loop", lambda: asyncio.sleep(0)
    )  # substitute a no-op coroutine factory
    await t._dial_and_start()

    assert t.descriptor_for_platform("discord") is None
    assert t._descriptor is None


# ───────────────────── adapter per-chat capability surface ─────────────────────


@pytest.mark.asyncio
async def test_adapter_resolves_per_chat_limits_from_inbound_platform():
    """The live bug shape: primary identity Telegram (4096), a Discord chat on
    the same adapter must get Discord's 2000 — not Telegram's scalar."""
    stub = MultiDescriptorStub(TELEGRAM, DISCORD)
    stub._identities = [("telegram", "bot-9"), ("discord", "app-1")]
    adapter = RelayAdapter(PlatformConfig(), TELEGRAM, transport=stub)
    await adapter.connect()

    await _push(stub, Platform.DISCORD, "dc-1")
    await _push(stub, Platform.TELEGRAM, "tg-1")

    # Scalar surface still the primary's (back-compat).
    assert adapter.MAX_MESSAGE_LENGTH == 4096
    # Per-chat: each chat resolves its own platform's cap.
    assert adapter.max_message_length_for_chat("dc-1") == 2000
    assert adapter.max_message_length_for_chat("tg-1") == 4096
    # Length unit follows the chat too: Telegram utf16, Discord codepoints.
    surrogate = "\U0001f600"  # 2 UTF-16 units, 1 codepoint
    assert adapter.message_len_fn_for_chat("tg-1")(surrogate) == 2
    assert adapter.message_len_fn_for_chat("dc-1")(surrogate) == 1


@pytest.mark.asyncio
async def test_adapter_unknown_chat_falls_back_to_scalar_descriptor():
    stub = MultiDescriptorStub(TELEGRAM, DISCORD)
    adapter = RelayAdapter(PlatformConfig(), TELEGRAM, transport=stub)
    await adapter.connect()
    # Never saw inbound for this chat — platform unknown -> scalar descriptor.
    assert adapter.max_message_length_for_chat("never-seen") == 4096


@pytest.mark.asyncio
async def test_adapter_transport_without_map_falls_back_to_scalar():
    """A plain StubConnector (no descriptor_for_platform) — e.g. an older or
    test transport — must keep the scalar behavior, not raise."""
    stub = StubConnector(TELEGRAM)
    adapter = RelayAdapter(PlatformConfig(), TELEGRAM, transport=stub)
    await adapter.connect()
    await _push(stub, Platform.DISCORD, "dc-1")
    assert adapter.max_message_length_for_chat("dc-1") == 4096


def test_native_adapter_defaults_scalar():
    """BasePlatformAdapter's default per-chat hooks mirror the scalar surface
    (native adapters are single-platform; nothing changes for them)."""
    from gateway.platforms.base import BasePlatformAdapter

    class _Native(BasePlatformAdapter):
        MAX_MESSAGE_LENGTH = 1234

        def __init__(self):  # bypass Base __init__ plumbing
            pass

        async def connect(self, *, is_reconnect: bool = False) -> bool:  # pragma: no cover
            return True

        async def disconnect(self) -> None:  # pragma: no cover
            pass

        async def send(self, chat_id, content, reply_to=None, metadata=None):  # pragma: no cover
            raise NotImplementedError

        async def get_chat_info(self, chat_id):  # pragma: no cover
            return {}

    a = _Native()
    assert a.max_message_length_for_chat("any") == 1234
    assert a.message_len_fn_for_chat("any") is a.message_len_fn


# ───────────────────── stream consumer integration ─────────────────────


@pytest.mark.asyncio
async def test_stream_consumer_raw_limit_uses_per_chat_cap():
    """_raw_message_limit resolves the CHAT's platform cap on a relay adapter:
    a Discord chat splits at 2000 even when the scalar descriptor says 4096.
    Without the fix this returned 4096 and a 2,543-char reply reached Discord
    whole -> HTTP 400."""
    from gateway.stream_consumer import GatewayStreamConsumer

    stub = MultiDescriptorStub(TELEGRAM, DISCORD)
    stub._identities = [("telegram", "bot-9"), ("discord", "app-1")]
    adapter = RelayAdapter(PlatformConfig(), TELEGRAM, transport=stub)
    await adapter.connect()
    await _push(stub, Platform.DISCORD, "dc-1")
    await _push(stub, Platform.TELEGRAM, "tg-1")

    dc = GatewayStreamConsumer.__new__(GatewayStreamConsumer)
    dc.adapter = adapter
    dc.chat_id = "dc-1"
    assert dc._raw_message_limit() == 2000

    tg = GatewayStreamConsumer.__new__(GatewayStreamConsumer)
    tg.adapter = adapter
    tg.chat_id = "tg-1"
    assert tg._raw_message_limit() == 4096
