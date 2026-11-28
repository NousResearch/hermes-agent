"""Per-chat message-editing capability on the relay connector path (#154674).

The bug class: ``RelayAdapter`` never adopted the descriptor's ``supports_edit``, so it inherited
``SUPPORTS_MESSAGE_EDITING = True`` and ``_build_stream_consumer_config`` streamed into every relay
chat. When the platform behind the chat cannot edit, the stream consumer's edit fallback re-sends
each update as a NEW message, so one long turn arrived as many separate texts (observed on an
SMS/iMessage connector). Editing is a per-CHAT fact: one socket fronts several platforms.

Covers:
  - the primary descriptor's ``supports_edit`` reaching the class-wide attribute (what every
    non-per-chat reader sees),
  - ``RelayAdapter.supports_edit_for_chat`` resolving the CHAT's descriptor on a connector that
    fronts an editable and a non-editable platform on one socket,
  - ``_build_stream_consumer_config`` skipping streaming for a non-editable chat and keeping the
    cursor for an editable one,
  - a chat that can native-DRAFT stream but cannot edit keeping its streaming lane: the draft IS
    the un-editable partial, so the skip rationale does not apply.
"""

from __future__ import annotations

import types
from typing import Optional

import pytest

from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource

from tests.gateway.relay.stub_connector import StubConnector


def _descriptor(
    platform: str,
    *,
    supports_edit: bool,
    supports_draft_streaming: bool = False,
    supported_ops: tuple = (),
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform=platform,
        label=platform.title(),
        max_message_length=4096,
        supports_draft_streaming=supports_draft_streaming,
        supports_edit=supports_edit,
        supports_threads=False,
        markdown_dialect="plain",
        len_unit="chars",
        supported_ops=supported_ops,
    )


EDITABLE = _descriptor("telegram", supports_edit=True)
SIGNAL = _descriptor("signal", supports_edit=False)
SIGNAL_DRAFTS = _descriptor(
    "signal", supports_edit=False, supports_draft_streaming=True, supported_ops=("send", "draft")
)


class MultiDescriptorStub(StubConnector):
    """StubConnector plus the per-platform descriptor map a multi-platform socket negotiates."""

    def __init__(self, primary: CapabilityDescriptor, *others: CapabilityDescriptor) -> None:
        super().__init__(primary)
        self._by_platform = {d.platform: d for d in (primary, *others)}

    def descriptor_for_platform(self, platform: str) -> Optional[CapabilityDescriptor]:
        return self._by_platform.get(platform)


async def _attach(stub: StubConnector, platform: Platform, chat_id: str) -> None:
    """Drive one inbound message so the adapter learns that chat's platform (as production does)."""
    await stub.push_inbound(
        MessageEvent(
            text="hi",
            message_type=MessageType.TEXT,
            source=SessionSource(platform=platform, chat_id=chat_id, chat_type="dm", user_id="u-1"),
        )
    )


def _stream_cfg(*, transport: str = "auto") -> StreamingConfig:
    return StreamingConfig(enabled=True, transport=transport)


def _source(chat_id: str) -> SessionSource:
    return SessionSource(platform=Platform.RELAY, chat_id=chat_id, chat_type="dm", user_id="u-1")


def _build(source: SessionSource, adapter: RelayAdapter, *, on_missing_cursor: str = "raise",
           transport: str = "auto"):
    # The method reads only its arguments (no runner state), so a bare stand-in self is faithful.
    return GatewayTurnMixin._build_stream_consumer_config(
        types.SimpleNamespace(), source, _stream_cfg(transport=transport), adapter,
        on_missing_cursor=on_missing_cursor
    )


# ── the adapter's class-wide answer ─────────────────────────────────────────────────────────────


def test_descriptor_edit_refusal_reaches_the_class_wide_attribute():
    """The class-wide read every non-per-chat consumer does (`getattr(adapter, …, True)`)."""
    assert getattr(RelayAdapter(PlatformConfig(), SIGNAL), "SUPPORTS_MESSAGE_EDITING", True) is False


def test_editable_descriptor_keeps_editing_enabled():
    assert RelayAdapter(PlatformConfig(), EDITABLE).SUPPORTS_MESSAGE_EDITING is True  # type: ignore[attr-defined]


# ── the per-chat answer on a multi-platform socket ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_supports_edit_for_chat_resolves_the_chats_own_descriptor():
    stub = MultiDescriptorStub(EDITABLE, SIGNAL)
    adapter = RelayAdapter(PlatformConfig(), EDITABLE, transport=stub)
    await adapter.connect()
    await _attach(stub, Platform.TELEGRAM, "tg-1")
    await _attach(stub, Platform.SIGNAL, "sig-1")

    assert adapter.supports_edit_for_chat("tg-1") is True
    assert adapter.supports_edit_for_chat("sig-1") is False
    # A chat never seen inbound keeps the primary's answer (the pre-existing scalar fallback).
    assert adapter.supports_edit_for_chat("unknown-chat") is True


# ── the streaming decision ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_non_editable_chat_skips_streaming():
    stub = MultiDescriptorStub(EDITABLE, SIGNAL)
    adapter = RelayAdapter(PlatformConfig(), EDITABLE, transport=stub)
    await adapter.connect()
    await _attach(stub, Platform.SIGNAL, "sig-1")

    with pytest.raises(RuntimeError):
        _build(_source("sig-1"), adapter)


@pytest.mark.asyncio
async def test_editable_chat_still_streams_with_the_cursor():
    stub = MultiDescriptorStub(EDITABLE, SIGNAL)
    adapter = RelayAdapter(PlatformConfig(), EDITABLE, transport=stub)
    await adapter.connect()
    await _attach(stub, Platform.SIGNAL, "sig-1")
    await _attach(stub, Platform.TELEGRAM, "tg-1")

    cfg, _pause = _build(_source("tg-1"), adapter)
    assert cfg.cursor == _stream_cfg().cursor


@pytest.mark.asyncio
async def test_draft_capable_chat_keeps_streaming_without_a_cursor():
    """A draft IS the un-editable partial, so a chat that can draft keeps its streaming lane."""
    stub = MultiDescriptorStub(EDITABLE, SIGNAL_DRAFTS)
    adapter = RelayAdapter(PlatformConfig(), EDITABLE, transport=stub)
    await adapter.connect()
    await _attach(stub, Platform.SIGNAL, "sig-1")

    cfg, _pause = _build(_source("sig-1"), adapter)
    assert cfg.cursor == ""

@pytest.mark.asyncio
async def test_explicit_edit_transport_cannot_stream_to_draft_only_chat():
    """Draft support is irrelevant when edit-only transport cannot select native drafts."""
    stub = MultiDescriptorStub(EDITABLE, SIGNAL_DRAFTS)
    adapter = RelayAdapter(PlatformConfig(), EDITABLE, transport=stub)
    await adapter.connect()
    await _attach(stub, Platform.SIGNAL, "sig-1")

    # 'auto' uses the native draft path; 'edit' uses send/edit and would fragment SMS.
    with pytest.raises(RuntimeError, match="skip streaming for non-editable platform"):
        _build(_source("sig-1"), adapter, transport="edit")
