"""RelayAdapter capability-advertisement tests (relay Phase 1, Task 1.1)."""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor


def make_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="telegram",
        label="Telegram",
        max_message_length=4096,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="markdown_v2",
        len_unit="utf16",
        emoji="\u2708\ufe0f",
        platform_hint="",
        pii_safe=False,
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


def _adapter(**desc_kw) -> RelayAdapter:
    return RelayAdapter(PlatformConfig(), make_desc(**desc_kw))


def test_relay_platform_member_exists():
    assert Platform("relay") is Platform.RELAY


def test_advertises_descriptor_max_length():
    a = _adapter(max_message_length=2000)
    assert a.MAX_MESSAGE_LENGTH == 2000


def test_supports_draft_streaming_follows_descriptor():
    assert _adapter(supports_draft_streaming=False).supports_draft_streaming() is False
    assert _adapter(supports_draft_streaming=True).supports_draft_streaming() is True


def test_len_fn_utf16_counts_code_units():
    a = _adapter(len_unit="utf16")
    # An astral-plane emoji is two UTF-16 code units.
    assert a.message_len_fn("\U0001f600") == 2


def test_len_fn_chars_uses_builtin_len():
    a = _adapter(len_unit="chars")
    assert a.message_len_fn("\U0001f600") == 1


def test_is_a_base_platform_adapter():
    # stream_consumer's isinstance(adapter, BasePlatformAdapter) guard must pass.
    from gateway.platforms.base import BasePlatformAdapter

    assert isinstance(_adapter(), BasePlatformAdapter)


@pytest.mark.asyncio
async def test_connect_without_transport_raises():
    a = _adapter()
    with pytest.raises(RuntimeError, match="no transport"):
        await a.connect()


@pytest.mark.asyncio
async def test_send_without_transport_returns_failure():
    a = _adapter()
    result = await a.send("chat1", "hello")
    assert result.success is False
    assert result.error == "no transport"


class _CaptureTransport:
    """Minimal RelayTransport stand-in that records the outbound action."""

    def __init__(self):
        self.sent = None

    def set_inbound_handler(self, h):  # noqa: D401
        self._h = h

    async def send_outbound(self, action):
        self.sent = action
        return {"success": True, "message_id": "m1"}


def _make_event(chat_id="chan-1", guild_id="guild-9"):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    src = SessionSource(
        platform=Platform.RELAY,
        chat_id=chat_id,
        chat_type="channel",
        guild_id=guild_id,
    )
    return MessageEvent(text="hi", source=src, message_type=MessageType.TEXT)


@pytest.mark.asyncio
async def test_send_reattaches_guild_id_from_inbound_scope():
    """The connector's egress guard resolves the owning tenant from
    metadata.guild_id; the gateway's generic delivery path drops it, so the
    relay adapter must re-attach the guild scope learned from the inbound event.
    Regression for live 'discord egress declined: target not routed to an
    onboarded tenant'."""
    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=t)
    # Simulate the connector delivering an inbound message in guild-9 / chan-1,
    # but don't run the full handle_message pipeline — just the scope capture.
    a._capture_scope(_make_event(chat_id="chan-1", guild_id="guild-9"))

    await a.send("chan-1", "the reply")

    assert t.sent["metadata"].get("guild_id") == "guild-9"


@pytest.mark.asyncio
async def test_send_without_known_scope_omits_guild_id():
    """A chat we never saw inbound (e.g. a DM) gets no guild_id — no-op, never
    invents a scope."""
    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=t)
    await a.send("unknown-chat", "hi")
    assert "guild_id" not in t.sent["metadata"]


@pytest.mark.asyncio
async def test_send_preserves_explicit_guild_id():
    """An explicitly-provided metadata.guild_id is never overwritten."""
    t = _CaptureTransport()
    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=t)
    a._capture_scope(_make_event(chat_id="chan-1", guild_id="guild-9"))
    await a.send("chan-1", "hi", metadata={"guild_id": "explicit-1"})
    assert t.sent["metadata"]["guild_id"] == "explicit-1"
