"""Internal wake entry points retain relay routing on cold adapters (#121001)."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.session import SessionSource
from gateway.wake import WakeNotAccepted, admit_internal_event, deliver_wake


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["admit", "deliver"])
@pytest.mark.parametrize("scope_id", [None, "guild-1"])
@pytest.mark.parametrize("busy", [False, True])
async def test_cold_internal_wake_reply_carries_origin(entry, scope_id, busy, monkeypatch):
    frames = []

    class Transport:
        async def send_outbound(self, action, *, platform=None):
            frames.append((action, platform))
            return {"success": True, "message_id": "reply-1"}

    adapter = RelayAdapter(PlatformConfig(), CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="discord", label="Discord",
        max_message_length=2000, supports_draft_streaming=False, supports_edit=True,
        supports_threads=True, markdown_dialect="markdown", len_unit="codepoints",
        emoji="", platform_hint="", pii_safe=False,
    ), transport=Transport())
    source = SessionSource(platform=Platform.DISCORD, chat_id="chat-1",
                           chat_type="group" if scope_id else "dm", user_id="user-1",
                           scope_id=scope_id, profile="worker")

    async def handle(event):
        event._gateway_accepted = True
        # Stand in for the model turn, not the real relay egress path.
        result = await adapter.send(event.source.chat_id, "wake reply")
        assert result.success

    event = MessageEvent(text="wake", source=source, message_type=MessageType.TEXT, internal=True)
    if busy:
        # Real BasePlatformAdapter admission/queue path, with a running-turn guard.
        adapter.set_message_handler(handle)
        key = adapter._event_session_key(event)
        adapter._active_sessions[key] = asyncio.Event()
    else:
        monkeypatch.setattr(adapter, "handle_message", handle)
    assert not adapter._dm_user_by_chat
    if entry == "admit":
        await admit_internal_event(adapter, event)
    else:
        await deliver_wake(adapter, text="wake", source=source)
    if busy:
        assert frames == []
        queued = adapter._pending_messages[key]
        assert queued.internal and queued.text == "wake"
        await handle(queued)
    action, platform = frames[0]
    assert action["metadata"].get("user_id") == "user-1"
    assert action["metadata"].get("profile") == "worker"
    assert action["metadata"].get("scope_id") == scope_id
    assert platform == "discord"
    assert adapter._with_scope("other-chat", None) == {}
    assert adapter._with_scope("chat-1", {"user_id": "explicit"})["user_id"] == "explicit"


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [True, False])
async def test_adapters_without_priming_keep_admission_contract(accepted):
    async def handle(event):
        event._gateway_accepted = accepted
    event = SimpleNamespace(_gateway_accepted=True)
    adapter = SimpleNamespace(handle_message=handle)
    if accepted:
        await admit_internal_event(adapter, event)
    else:
        with pytest.raises(WakeNotAccepted):
            await admit_internal_event(adapter, event)


@pytest.mark.asyncio
async def test_priming_failure_is_not_admitted():
    calls = []
    def prime(event):
        raise RuntimeError("routing unavailable")
    async def handle(event):
        calls.append(event)
    event = SimpleNamespace(_gateway_accepted=True)
    adapter = SimpleNamespace(handle_message=handle, prime_routing_cache=prime)
    with pytest.raises(RuntimeError, match="routing unavailable"):
        await admit_internal_event(adapter, event)
    assert event._gateway_accepted is False
    assert calls == []
