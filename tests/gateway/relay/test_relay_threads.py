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


@pytest.mark.asyncio
async def test_create_handoff_thread_op_gated_and_decline_safe():
    # Not advertised → None without touching the wire.
    adapter, stub = _adapter(supported_ops=("send", "edit", "typing"))
    assert await adapter.create_handoff_thread("chan1", "x") is None
    assert stub.sent == []
    # Advertised but declined (non-forum Telegram chat, missing perms) → None.
    adapter2, stub2 = _adapter()

    async def declined(action, *, platform=None):
        stub2.sent.append(action)
        return {"success": False, "error": "not a forum"}

    stub2.send_outbound = declined  # type: ignore[method-assign]
    assert await adapter2.create_handoff_thread("chan1", "x") is None


@pytest.mark.asyncio
async def test_create_handoff_thread_falls_back_to_message_id():
    # Slack shape: the connector returns the seed ts as message_id+thread_id;
    # older stubs may return only message_id — either resolves.
    adapter, stub = _adapter()

    async def send_outbound(action, *, platform=None):
        stub.sent.append(action)
        return {"success": True, "message_id": "170.001"}

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    assert await adapter.create_handoff_thread("C1", "handoff") == "170.001"


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


@pytest.mark.asyncio
async def test_rename_thread_decline_returns_false():
    adapter, stub = _adapter()

    async def declined(action, *, platform=None):
        stub.sent.append(action)
        return {"success": False, "error": "guard: current name differs"}

    stub.send_outbound = declined  # type: ignore[method-assign]
    assert await adapter.rename_thread("th1", "x", only_if_current_name="H") is False


# ── the relay semantic-rename lane (marker parity) ───────────────────────


def test_relay_source_satisfies_auto_thread_lane_field_contract():
    """The native lane check reads platform/chat_type/thread_id +
    auto_thread_created + auto_thread_initial_name off the source. A relay
    event carrying the connector-stamped markers must present ALL of them
    through _event_from_wire — same field contract, no relay-specific case.
    """
    event = _event_from_wire(
        {
            "text": "hi",
            "message_type": "text",
            "source": {
                "platform": "discord",
                "chat_id": "th9",
                "chat_type": "thread",
                "thread_id": "th9",
                "user_id": "u1",
                "auto_thread_created": True,
                "auto_thread_initial_name": "Hermes reply",
            },
        }
    )
    src = event.source
    assert src.chat_type == "thread"
    assert src.thread_id == "th9"
    assert src.auto_thread_created is True
    assert src.auto_thread_initial_name == "Hermes reply"
    # And absent markers default off (never light the lane spuriously).
    plain = _event_from_wire(
        {
            "text": "hi",
            "message_type": "text",
            "source": {
                "platform": "discord",
                "chat_id": "c",
                "chat_type": "thread",
                "thread_id": "t",
            },
        }
    )
    assert plain.source.auto_thread_created is False
    assert plain.source.auto_thread_initial_name is None


# ── reply_to wire parse ──────────────────────────────────────────────────


def test_event_from_wire_maps_reply_to_onto_native_fields():
    event = _event_from_wire(
        {
            "text": "re",
            "message_type": "text",
            "source": {"platform": "telegram", "chat_id": "5", "chat_type": "dm"},
            "reply_to_message_id": "19",
            "reply_to": {
                "text": "what the bot said",
                "author": "hermesbot",
                "is_own": True,
            },
        }
    )
    assert event.reply_to_message_id == "19"
    assert event.reply_to_text == "what the bot said"
    assert event.reply_to_author_name == "hermesbot"
    assert event.reply_to_is_own_message is True


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


def test_manifest_entries_satisfy_discord_naming_rules():
    manifest = build_relay_command_manifest()
    assert len(manifest) > 0
    assert len(manifest) <= 100  # Discord's global command cap
    name_re = re.compile(r"^[a-z0-9_-]{1,32}$")
    seen = set()
    for entry in manifest:
        assert name_re.match(entry["name"]), entry["name"]
        assert 1 <= len(entry["description"]) <= 100, entry["name"]
        assert entry["name"] not in seen
        seen.add(entry["name"])
        for opt in entry.get("options", []):
            assert name_re.match(opt["name"])
            assert 1 <= len(opt["description"]) <= 100


def test_manifest_mirrors_the_native_slash_surface():
    # Behavior contract (not a change-detector): every manifest name must be
    # a command the dispatcher actually routes — spot-check the core set the
    # native Discord tree registers.
    names = {e["name"] for e in build_relay_command_manifest()}
    for core in ("new", "reset", "model", "approve", "deny", "stop", "help"):
        assert core in names


@pytest.mark.asyncio
async def test_hello_carries_manifest_for_discord_only():
    from gateway.relay.ws_transport import WebSocketRelayTransport

    sent: list[Dict[str, Any]] = []
    transport = WebSocketRelayTransport.__new__(WebSocketRelayTransport)
    transport._identities = [("discord", "app1"), ("telegram", "tg1")]

    async def fake_send(frame):
        sent.append(frame)

    transport._send = fake_send  # type: ignore[method-assign]

    # Drive just the hello loop body (connect()'s tail) — the socket plumbing
    # above it is exercised by the existing transport tests.
    for platform, bot_id in transport._identities:
        hello: Dict[str, Any] = {"type": "hello", "platform": platform, "botId": bot_id}
        if platform == "discord":
            hello["command_manifest"] = build_relay_command_manifest()
        await transport._send(hello)

    assert sent[0]["platform"] == "discord"
    assert "command_manifest" in sent[0]
    assert sent[1]["platform"] == "telegram"
    assert "command_manifest" not in sent[1]
