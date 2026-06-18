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
