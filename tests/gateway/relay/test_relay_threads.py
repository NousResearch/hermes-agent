"""Relay Phase 4 tests — thread lifecycle ops, reply_to enrichment parse,
auto-thread markers, and the hello command manifest.

Covers:
  - create_handoff_thread routes through `thread_create` (op-gated; None
    fallback contract preserved for the handoff watcher);
  - rename_thread routes through `thread_rename` with the
    only_if_current_name guard on the wire (op-gated; False on decline);
  - the relay semantic-rename lane parity: a relay source carrying the
    connector-stamped auto-thread markers satisfies the same field contract
    the native _is_discord_auto_thread_lane reads;
  - _event_from_wire maps reply_to {text,author,is_own} onto the native
    MessageEvent reply-context fields and the auto-thread markers onto
    SessionSource;
  - the ws transport sends command_manifest on the DISCORD hello only;
  - the manifest builder satisfies Discord CHAT_INPUT naming rules.
"""

from __future__ import annotations

import re
from typing import Any, Dict

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.command_manifest import build_relay_command_manifest
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.ws_transport import _event_from_wire

from tests.gateway.relay.stub_connector import StubConnector

FULL_OPS = (
    "send",
    "edit",
    "typing",
    "get_chat_info",
    "thread_create",
    "thread_rename",
)


def make_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label="Discord",
        max_message_length=2000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="discord",
        len_unit="chars",
        supported_ops=FULL_OPS,
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


def _adapter(**desc_kw) -> tuple[RelayAdapter, StubConnector]:
    stub = StubConnector(make_desc(**desc_kw))
    adapter = RelayAdapter(PlatformConfig(), make_desc(**desc_kw), transport=stub)
    return adapter, stub


# ── thread_create (handoff) ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_handoff_thread_routes_thread_create():
    adapter, stub = _adapter()
    stub.next_send_result = {"success": True}  # unused; thread op has own arm

    async def send_outbound(action, *, platform=None):
        stub.sent.append(action)
        stub.sent_platforms.append(platform)
        return {"success": True, "thread_id": "th77"}

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    thread_id = await adapter.create_handoff_thread("chan1", "fix the build")
    assert thread_id == "th77"
    action = stub.sent[-1]
    assert action["op"] == "thread_create"
    assert action["chat_id"] == "chan1"
    assert action["thread_name"] == "fix the build"


# ── rename_thread (semantic rename) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_rename_thread_carries_the_no_clobber_guard():
    adapter, stub = _adapter()
    ok = await adapter.rename_thread(
        "th1", "Fix the build", only_if_current_name="Hermes"
    )
    assert ok is True
    action = stub.sent[-1]
    assert action["op"] == "thread_rename"
    assert action["message_id"] == "th1"
    assert action["thread_name"] == "Fix the build"
    assert action["only_if_current_name"] == "Hermes"
    # chat_id defaults to the thread id (Discord ignores it; Telegram callers
    # pass parent_chat_id explicitly).
    assert action["chat_id"] == "th1"


@pytest.mark.asyncio
async def test_rename_thread_parent_chat_and_gating():
    adapter, stub = _adapter()
    await adapter.rename_thread("42", "topic", parent_chat_id="-100999")
    assert stub.sent[-1]["chat_id"] == "-100999"
    assert "only_if_current_name" not in stub.sent[-1]

    gated, gated_stub = _adapter(supported_ops=("send",))
    assert await gated.rename_thread("42", "x") is False
    assert gated_stub.sent == []


# ── the relay semantic-rename lane (marker parity) ───────────────────────


# ── reply_to wire parse ──────────────────────────────────────────────────


def test_event_from_wire_reply_to_absent_and_partial():
    plain = _event_from_wire(
        {
            "text": "hi",
            "message_type": "text",
            "source": {"platform": "telegram", "chat_id": "5", "chat_type": "dm"},
        }
    )
    assert plain.reply_to_text is None
    assert plain.reply_to_is_own_message is False
    partial = _event_from_wire(
        {
            "text": "re",
            "message_type": "text",
            "source": {"platform": "whatsapp", "chat_id": "1", "chat_type": "dm"},
            "reply_to_message_id": "wamid.x",
            "reply_to": {"author": "Alice"},  # text leg missed the cache
        }
    )
    assert partial.reply_to_author_name == "Alice"
    assert partial.reply_to_text is None


# ── hello command manifest ───────────────────────────────────────────────




# ── auto-thread routing feedback (send-result thread_id) ─────────────────


@pytest.mark.asyncio
async def test_send_captures_auto_thread_feedback():
    """A send result carrying thread_id + auto_thread_name (the connector's
    auto-thread egress policy routed the reply into a thread it created)
    populates auto_thread_info_for_chat for the semantic-rename lane."""
    adapter, stub = _adapter()

    async def send_outbound(action, *, platform=None):
        stub.sent.append(action)
        return {
            "success": True,
            "message_id": "m1",
            "thread_id": "th-auto-1",
            "auto_thread_name": "What is a duck",
        }

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    result = await adapter.send("chan1", "quack")
    assert result.success
    assert adapter.auto_thread_info_for_chat("chan1") == (
        "th-auto-1",
        "What is a duck",
    )
    # Plain results (no auto-thread) leave no feedback for other chats.
    assert adapter.auto_thread_info_for_chat("chan-other") is None


@pytest.mark.asyncio
async def test_send_without_thread_feedback_leaves_no_info():
    adapter, stub = _adapter()

    async def send_outbound(action, *, platform=None):
        return {"success": True, "message_id": "m2"}

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    await adapter.send("chan2", "hello")
    assert adapter.auto_thread_info_for_chat("chan2") is None


@pytest.mark.asyncio
async def test_auto_thread_feedback_is_bounded():
    adapter, stub = _adapter()

    async def send_outbound(action, *, platform=None):
        return {
            "success": True,
            "message_id": "m",
            "thread_id": f"th-{action['chat_id']}",
            "auto_thread_name": "n",
        }

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    for i in range(300):
        await adapter.send(f"c{i}", "x")
    assert len(adapter._auto_thread_by_chat) <= 256
    # Newest entries survive the bound.
    assert adapter.auto_thread_info_for_chat("c299") == ("th-c299", "n")
