"""Regression tests for WhatsApp busy-session steer mode.

Covers two UX issues reported in WhatsApp:
1) When the agent is running, additional user messages could be dropped because
   BasePlatformAdapter overwrote _pending_messages[session_key].
2) /steer sent mid-run did not dispatch because it wasn't a bypass command and
   therefore hit the active-session guard.

This test suite validates that WhatsAppAdapter can enable a busy-session handler
that dispatches /steer inline.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import build_session_key


@pytest.mark.asyncio
async def test_whatsapp_busy_text_policy_default_is_upsert_steer_but_adapter_does_not_recurse():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True))
    assert getattr(adapter, "_busy_text_policy", None) == "upsert_steer"
    assert getattr(adapter, "_busy_session_handler", None) is None


@pytest.mark.asyncio
async def test_whatsapp_busy_text_policy_steer_waits_for_runner_handler():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"busy_text_policy": "steer"}))
    assert getattr(adapter, "_busy_text_policy", None) == "steer"
    assert getattr(adapter, "_busy_session_handler", None) is None


@pytest.mark.asyncio
async def test_whatsapp_busy_handler_converts_plain_text_to_steer_and_sends_ack():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"busy_text_policy": "steer"}))

    seen = {}

    async def handler(evt: MessageEvent):
        # Capture what the adapter dispatched.
        seen["text"] = evt.text
        return "ok"

    adapter.set_message_handler(handler)

    adapter._send_with_retry = AsyncMock(return_value=None)  # avoid HTTP

    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    evt = MessageEvent(text="focus on the error", message_type=MessageType.TEXT, source=src, message_id="m1")

    # Call handler directly (BasePlatformAdapter would call it only when busy)
    handled = await adapter._handle_busy_text_as_steer(evt, session_key="sess")
    assert handled is True
    assert seen["text"].startswith("/steer ")
    assert "focus on the error" in seen["text"]


@pytest.mark.asyncio
async def test_whatsapp_busy_handler_dispatches_explicit_steer_command_inline():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"busy_text_policy": "steer"}))

    seen = {}

    async def handler(evt: MessageEvent):
        seen["text"] = evt.text
        return "queued"

    adapter.set_message_handler(handler)
    adapter._send_with_retry = AsyncMock(return_value=None)  # avoid HTTP

    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    evt = MessageEvent(
        text="/steer keep the current task running; just add this context",
        message_type=MessageType.COMMAND,
        source=src,
        message_id="m-cmd",
    )

    handled = await adapter._handle_busy_text_as_steer(evt, session_key="sess")
    assert handled is True
    assert seen["text"] == "/steer keep the current task running; just add this context"


@pytest.mark.asyncio
async def test_whatsapp_handle_message_dispatches_explicit_steer_while_busy_without_queueing():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"busy_text_policy": "steer", "enable_sessions": False}))

    seen = {}

    async def handler(evt: MessageEvent):
        seen["text"] = evt.text
        return "queued"

    adapter.set_message_handler(handler)
    adapter._send_with_retry = AsyncMock(return_value=None)  # avoid HTTP

    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    session_key = build_session_key(src)
    interrupt_event = asyncio.Event()
    adapter._active_sessions[session_key] = interrupt_event

    evt = MessageEvent(
        text="/steer do not stop; just fold this into the current task",
        message_type=MessageType.COMMAND,
        source=src,
        message_id="m-active-cmd",
    )

    await adapter.handle_message(evt)

    assert seen["text"] == "/steer do not stop; just fold this into the current task"
    assert session_key not in adapter._pending_messages
    assert interrupt_event.is_set() is False


@pytest.mark.asyncio
async def test_whatsapp_build_event_maps_quoted_fields_to_reply_context():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True))
    data = {
        "messageId": "m2",
        "chatId": "123@lid",
        "senderId": "123@lid",
        "senderName": "Seb",
        "chatName": "Seb",
        "isGroup": False,
        "body": "yep",
        "hasMedia": False,
        "mediaUrls": [],
        "quotedMessageId": "q1",
        "quotedText": "the thing you said earlier",
    }

    event = await adapter._build_message_event(data)
    assert event is not None
    assert event.reply_to_message_id == "q1"
    assert event.reply_to_text == "the thing you said earlier"


def test_gateway_runner_installs_whatsapp_upsert_steer_handler_without_adapter_recursion():
    """Gateway owns WhatsApp upsert-steer so adapter never recursively calls the gateway."""
    from gateway.platforms.whatsapp import WhatsAppAdapter
    from gateway.run import GatewayRunner

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"busy_text_policy": "upsert_steer"}))
    assert getattr(adapter, "_busy_session_handler", None) is None

    runner = object.__new__(GatewayRunner)
    runner._install_default_busy_session_handler(adapter)

    assert getattr(adapter, "_busy_session_handler", None) == runner._handle_active_session_upsert_steer


@pytest.mark.asyncio
async def test_gateway_runner_upsert_steer_updates_running_agent_without_message_handler():
    from gateway.config import Platform
    from gateway.platforms.whatsapp import WhatsAppAdapter
    from gateway.run import GatewayRunner

    class FakeAgent:
        def __init__(self):
            self.seen = []
        def steer(self, text):
            self.seen.append(text)
            return True

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"busy_text_policy": "upsert_steer"}))
    adapter.send = AsyncMock(return_value=None)
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    session_key = build_session_key(src)
    agent = FakeAgent()

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner._running_agents = {session_key: agent}

    evt = MessageEvent(text="fold this into the current task", message_type=MessageType.TEXT, source=src, message_id="m-upsert")

    handled = await runner._handle_active_session_upsert_steer(evt, session_key)

    assert handled is True
    assert agent.seen == ["fold this into the current task"]
    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[1].startswith("🕹️ /steer queued into the current session")


@pytest.mark.asyncio
async def test_gateway_runner_upsert_steer_unknown_slash_message_returns_command_guidance():
    from gateway.config import Platform
    from gateway.platforms.whatsapp import WhatsAppAdapter
    from gateway.run import GatewayRunner

    class FakeAgent:
        def __init__(self):
            self.seen = []
        def steer(self, text):
            self.seen.append(text)
            return True

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"busy_text_policy": "upsert_steer"}))
    adapter.send = AsyncMock(return_value=None)
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    session_key = build_session_key(src)
    agent = FakeAgent()

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner._running_agents = {session_key: agent}
    runner._draining = False

    evt = MessageEvent(text="/steeer keep going", message_type=MessageType.COMMAND, source=src, message_id="m-typo")

    handled = await runner._handle_active_session_upsert_steer(evt, session_key)

    assert handled is True
    assert agent.seen == []
    adapter.send.assert_awaited_once()
    assert "Unknown command" in adapter.send.await_args.args[1]
    assert "/steeer" in adapter.send.await_args.args[1]
    assert "/commands" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_gateway_runner_upsert_steer_inserts_when_no_running_agent():
    from gateway.config import Platform
    from gateway.platforms.whatsapp import WhatsAppAdapter
    from gateway.run import GatewayRunner

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"busy_text_policy": "upsert_steer"}))
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    session_key = build_session_key(src)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner._running_agents = {}

    evt = MessageEvent(text="start this normally", message_type=MessageType.TEXT, source=src, message_id="m-insert")

    handled = await runner._handle_active_session_upsert_steer(evt, session_key)

    assert handled is False


@pytest.mark.asyncio
async def test_gateway_runner_upsert_steer_explicit_policy_only_updates_slash_steer():
    from gateway.config import Platform
    from gateway.platforms.whatsapp import WhatsAppAdapter
    from gateway.run import GatewayRunner

    class FakeAgent:
        def steer(self, text):
            raise AssertionError("plain text should not auto-steer in explicit_steer mode")

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"busy_text_policy": "explicit_steer"}))
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    session_key = build_session_key(src)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner._running_agents = {session_key: FakeAgent()}

    evt = MessageEvent(text="plain follow-up", message_type=MessageType.TEXT, source=src, message_id="m-explicit")

    handled = await runner._handle_active_session_upsert_steer(evt, session_key)

    assert handled is False


def test_whatsapp_session_router_reuses_latest_visible_session_for_unquoted_messages():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"enable_sessions": True}))
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")

    first = MessageEvent(text="start topic", message_type=MessageType.TEXT, source=src, message_id="m1")
    routed_first = adapter._route_event_to_session(first)
    assert routed_first.source.thread_id

    second = MessageEvent(text="follow up", message_type=MessageType.TEXT, source=src, message_id="m2")
    routed_second = adapter._route_event_to_session(second)

    assert routed_second.source.thread_id == routed_first.source.thread_id
    assert build_session_key(routed_second.source).endswith(routed_first.source.thread_id)


def test_whatsapp_session_router_creates_new_session_for_unknown_quoted_reply():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"enable_sessions": True}))
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    focused = adapter._route_event_to_session(
        MessageEvent(text="current topic", message_type=MessageType.TEXT, source=src, message_id="current")
    )

    quoted = MessageEvent(
        text="about that older thing",
        message_type=MessageType.TEXT,
        source=src,
        message_id="m-quoted",
        reply_to_message_id="unknown-old-message",
        reply_to_text="old quoted text",
    )
    routed = adapter._route_event_to_session(quoted)

    assert routed.source.thread_id != focused.source.thread_id
    assert routed.reply_to_text == "old quoted text"
    assert adapter._whatsapp_message_sessions["unknown-old-message"] == routed.source.thread_id


def test_whatsapp_session_router_routes_known_quoted_reply_to_original_session():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"enable_sessions": True}))
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    session_a = adapter._route_event_to_session(
        MessageEvent(text="topic a", message_type=MessageType.TEXT, source=src, message_id="a1")
    ).source.thread_id
    adapter._register_whatsapp_session_message("assistant-a", session_a)
    session_b = adapter._route_event_to_session(
        MessageEvent(
            text="topic b",
            message_type=MessageType.TEXT,
            source=src,
            message_id="b1",
            reply_to_message_id="unknown-b",
            reply_to_text="seed b",
        )
    ).source.thread_id
    assert session_b != session_a

    routed = adapter._route_event_to_session(
        MessageEvent(
            text="back to a",
            message_type=MessageType.TEXT,
            source=src,
            message_id="a2",
            reply_to_message_id="assistant-a",
            reply_to_text="assistant answer in topic a",
        )
    )

    assert routed.source.thread_id == session_a


def test_whatsapp_session_router_preserves_explicit_thread_id_for_internal_events():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"enable_sessions": True}))
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    session_a = adapter._route_event_to_session(
        MessageEvent(text="topic a", message_type=MessageType.TEXT, source=src, message_id="a1")
    ).source.thread_id
    session_b = adapter._route_event_to_session(
        MessageEvent(
            text="topic b",
            message_type=MessageType.TEXT,
            source=src,
            message_id="b1",
            reply_to_message_id="unknown-b",
            reply_to_text="seed b",
        )
    ).source.thread_id
    assert session_b != session_a

    # Simulates a background-process completion/watch notification whose
    # watcher stored the original session id. Routing must not silently retarget
    # it to the currently focused session.
    internal_src = adapter.build_source(
        chat_id="123@lid",
        chat_type="dm",
        user_id="123@lid",
        user_name="Seb",
        thread_id=session_a,
    )
    routed = adapter._route_event_to_session(
        MessageEvent(
            text="[SYSTEM: Background process completed]",
            message_type=MessageType.TEXT,
            source=internal_src,
            internal=True,
        )
    )

    assert routed.source.thread_id == session_a


def test_whatsapp_session_router_expires_idle_focus_session_for_unquoted_messages():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"enable_sessions": True, "session_idle_minutes": 60}))
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    first = adapter._route_event_to_session(
        MessageEvent(text="old topic", message_type=MessageType.TEXT, source=src, message_id="old")
    )
    old_session = first.source.thread_id
    adapter._whatsapp_sessions[old_session].last_activity_at = time.time() - (61 * 60)

    routed = adapter._route_event_to_session(
        MessageEvent(text="new after idle", message_type=MessageType.TEXT, source=src, message_id="new")
    )

    assert routed.source.thread_id != old_session


@pytest.mark.asyncio
async def test_whatsapp_sessions_command_lists_active_sessions_without_calling_agent():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"enable_sessions": True}))
    adapter.set_message_handler(AsyncMock(return_value="should not run"))
    adapter.send = AsyncMock(return_value=None)
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    session_a = adapter._route_event_to_session(
        MessageEvent(text="first topic", message_type=MessageType.TEXT, source=src, message_id="a1")
    ).source.thread_id
    session_b = adapter._route_event_to_session(
        MessageEvent(
            text="second topic",
            message_type=MessageType.TEXT,
            source=src,
            message_id="b1",
            reply_to_message_id="unknown-b",
            reply_to_text="second seed",
        )
    ).source.thread_id

    await adapter.handle_message(MessageEvent(text="/sessions", message_type=MessageType.COMMAND, source=src, message_id="cmd-list"))

    adapter._message_handler.assert_not_awaited()
    sent = adapter.send.await_args.args[1]
    assert "Active WhatsApp sessions" in sent
    assert "1." in sent and "2." in sent
    assert session_b in sent
    assert session_a in sent


@pytest.mark.asyncio
async def test_whatsapp_legacy_session_command_word_is_not_supported():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"enable_sessions": True}))
    adapter.set_message_handler(AsyncMock(return_value="ok"))
    adapter.send = AsyncMock(return_value=None)
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True))
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")

    await adapter.handle_message(MessageEvent(text="/" + "la" + "nes", message_type=MessageType.COMMAND, source=src, message_id="cmd-old"))
    if adapter._background_tasks:
        await asyncio.gather(*adapter._background_tasks)

    adapter._message_handler.assert_awaited_once()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_whatsapp_session_new_command_moves_focus_without_calling_agent():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"enable_sessions": True}))
    adapter.set_message_handler(AsyncMock(return_value="should not run"))
    adapter.send = AsyncMock(return_value=None)
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    old_session = adapter._route_event_to_session(
        MessageEvent(text="old topic", message_type=MessageType.TEXT, source=src, message_id="old")
    ).source.thread_id

    await adapter.handle_message(MessageEvent(text="/sessions new", message_type=MessageType.COMMAND, source=src, message_id="cmd-new"))
    routed = adapter._route_event_to_session(
        MessageEvent(text="now on forced new session", message_type=MessageType.TEXT, source=src, message_id="after-new")
    )

    adapter._message_handler.assert_not_awaited()
    assert routed.source.thread_id != old_session
    assert "New WhatsApp session" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_whatsapp_session_new_command_with_message_sends_immediate_session_notice():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"enable_sessions": True}))
    adapter.set_message_handler(AsyncMock(return_value="ok"))
    adapter.send = AsyncMock(return_value=None)
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")

    await adapter.handle_message(
        MessageEvent(text="/sessions new side quest", message_type=MessageType.COMMAND, source=src, message_id="cmd-new-msg")
    )

    adapter.send.assert_awaited_once()
    sent = adapter.send.await_args.args[1]
    assert "🧵 New WhatsApp session" in sent
    assert "side quest" in sent


@pytest.mark.asyncio
async def test_whatsapp_auto_new_session_sends_immediate_session_notice_when_reply_starts_side_session():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"enable_sessions": True}))
    adapter.set_message_handler(AsyncMock(return_value="ok"))
    adapter.send = AsyncMock(return_value=None)
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    first = adapter._route_event_to_session(
        MessageEvent(text="active topic", message_type=MessageType.TEXT, source=src, message_id="active-root")
    )

    await adapter.handle_message(
        MessageEvent(
            text="separate side topic",
            message_type=MessageType.TEXT,
            source=src,
            message_id="side-root",
            reply_to_message_id="unknown-side",
            reply_to_text="old side seed",
        )
    )

    adapter.send.assert_awaited_once()
    sent = adapter.send.await_args.args[1]
    assert "🧵 New WhatsApp session" in sent
    assert "separate side topic" in sent
    assert adapter.send.await_args.kwargs["reply_to"] == "side-root"
    assert adapter.send.await_args.kwargs["metadata"]["thread_id"] != first.source.thread_id


@pytest.mark.asyncio
async def test_whatsapp_session_number_command_switches_focus_to_listed_session():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"enable_sessions": True}))
    adapter.set_message_handler(AsyncMock(return_value="should not run"))
    adapter.send = AsyncMock(return_value=None)
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    session_a = adapter._route_event_to_session(
        MessageEvent(text="first topic", message_type=MessageType.TEXT, source=src, message_id="a1")
    ).source.thread_id
    session_b = adapter._route_event_to_session(
        MessageEvent(
            text="second topic",
            message_type=MessageType.TEXT,
            source=src,
            message_id="b1",
            reply_to_message_id="unknown-b",
            reply_to_text="second seed",
        )
    ).source.thread_id
    assert session_b != session_a

    await adapter.handle_message(MessageEvent(text="/sessions 2", message_type=MessageType.COMMAND, source=src, message_id="cmd-switch"))
    routed = adapter._route_event_to_session(
        MessageEvent(text="back on selected session", message_type=MessageType.TEXT, source=src, message_id="after-switch")
    )

    adapter._message_handler.assert_not_awaited()
    assert routed.source.thread_id == session_a
    assert "Switched WhatsApp session" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_whatsapp_handle_message_routes_different_session_without_interrupting_active_focus():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"enable_sessions": True, "busy_text_policy": "explicit_steer"}))
    adapter.set_message_handler(AsyncMock(return_value="ok"))
    adapter._send_with_retry = AsyncMock(return_value=None)
    src = adapter.build_source(chat_id="123@lid", chat_type="dm", user_id="123@lid", user_name="Seb")
    active = adapter._route_event_to_session(
        MessageEvent(text="active topic", message_type=MessageType.TEXT, source=src, message_id="active-root")
    )
    active_key = build_session_key(active.source)
    adapter._active_sessions[active_key] = asyncio.Event()

    side = MessageEvent(
        text="separate side topic",
        message_type=MessageType.TEXT,
        source=src,
        message_id="side-root",
        reply_to_message_id="unknown-side",
        reply_to_text="old side seed",
    )
    await adapter.handle_message(side)

    assert adapter._active_sessions[active_key].is_set() is False
    assert "side-root" in adapter._whatsapp_message_sessions
    assert adapter._whatsapp_message_sessions["side-root"] != active.source.thread_id


def test_gateway_runner_installs_default_busy_handler_when_adapter_has_none():
    """Adapters without a specialized handler still get the generic interrupt/queue handler."""
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    class DummyAdapter(BasePlatformAdapter):
        async def connect(self):
            return True

        async def disconnect(self):
            pass

        async def send_message(self, chat_id, content, reply_to=None, metadata=None):
            return True

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return True

        async def get_chat_info(self, chat_id):
            return {}

        async def send_typing(self, chat_id, metadata=None):
            pass

        async def mark_read(self, chat_id, message_id, metadata=None):
            pass

    adapter = DummyAdapter(PlatformConfig(enabled=True), Platform.API_SERVER)
    assert getattr(adapter, "_busy_session_handler", None) is None

    runner = object.__new__(GatewayRunner)
    runner._install_default_busy_session_handler(adapter)

    assert getattr(adapter, "_busy_session_handler", None) is not None
