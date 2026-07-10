"""Tests verifying interrupt key consistency between adapter and gateway.

Regression test for a bug where monitor_for_interrupt() in _run_agent used
source.chat_id to query the adapter, but the adapter stores interrupts under
the full session key (build_session_key output).  This mismatch meant
interrupts were never detected, causing subagents to ignore new messages.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionBinding, SessionSource, build_session_key


class StubAdapter(BasePlatformAdapter):
    """Minimal adapter for interrupt tests."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _source(chat_id="123456", chat_type="dm", thread_id=None):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
        thread_id=thread_id,
    )


class TestInterruptKeyConsistency:
    """Ensure adapter interrupt methods are queried with session_key, not chat_id."""

    def test_session_key_differs_from_chat_id_for_dm(self):
        """Session key for a DM is namespaced and includes the DM chat_id."""
        source = _source("123456", "dm")
        session_key = build_session_key(source)
        assert session_key != source.chat_id
        assert session_key == "agent:main:telegram:dm:123456"

    def test_session_key_differs_from_chat_id_for_group(self):
        """Session key for a group chat includes prefix, unlike raw chat_id."""
        source = _source("-1001234", "group")
        session_key = build_session_key(source)
        assert session_key != source.chat_id
        assert "agent:main:" in session_key
        assert source.chat_id in session_key

    @pytest.mark.asyncio
    async def test_has_pending_interrupt_requires_session_key(self):
        """has_pending_interrupt returns True only when queried with session_key."""
        adapter = StubAdapter()
        source = _source("123456", "dm")
        session_key = build_session_key(source)

        # Simulate adapter storing interrupt under session_key
        interrupt_event = asyncio.Event()
        adapter._active_sessions[session_key] = interrupt_event
        interrupt_event.set()

        # Using session_key → found
        assert adapter.has_pending_interrupt(session_key) is True

        # Using chat_id → NOT found (this was the bug)
        assert adapter.has_pending_interrupt(source.chat_id) is False

    @pytest.mark.asyncio
    async def test_get_pending_message_requires_session_key(self):
        """get_pending_message returns the event only with session_key."""
        adapter = StubAdapter()
        source = _source("123456", "dm")
        session_key = build_session_key(source)

        event = MessageEvent(text="hello", source=source, message_id="42")
        adapter._pending_messages[session_key] = event

        # Using chat_id → None (the bug)
        assert adapter.get_pending_message(source.chat_id) is None

        # Using session_key → found
        result = adapter.get_pending_message(session_key)
        assert result is event

    @pytest.mark.asyncio
    async def test_handle_message_stores_under_session_key(self):
        """handle_message stores pending messages under session_key, not chat_id."""
        adapter = StubAdapter()
        adapter._busy_text_mode = ""
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        source = _source("-1001234", "group")
        session_key = build_session_key(source)

        # Mark session as active
        adapter._active_sessions[session_key] = asyncio.Event()

        # Send a second message while session is active
        event = MessageEvent(text="interrupt!", source=source, message_id="2")
        await adapter.handle_message(event)

        # Stored under session_key
        assert session_key in adapter._pending_messages
        # NOT stored under chat_id
        assert source.chat_id not in adapter._pending_messages

        # Text follow-ups queue silently and do not interrupt the active turn.
        assert adapter._active_sessions[session_key].is_set() is False

    @pytest.mark.asyncio
    async def test_stop_resolves_session_binding_before_active_session_guard(self, monkeypatch):
        adapter = StubAdapter()
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        source = _source("-1001234", "group", thread_id="75911")
        binding = SessionBinding(namespace="gh-review", key="pull-request-7816")
        bound_source = _source("-1001234", "group", thread_id="75911")
        bound_source.session_binding = binding
        bound_session_key = build_session_key(bound_source)
        adapter._active_sessions[bound_session_key] = asyncio.Event()

        dispatched = []
        started = []

        async def dispatch_active_session_command(event, session_key, command):
            dispatched.append((session_key, command))

        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda name, **kwargs: [{"session_binding": binding}]
            if name == "pre_session_binding"
            else [],
        )
        monkeypatch.setattr(
            adapter,
            "_dispatch_active_session_command",
            dispatch_active_session_command,
        )
        monkeypatch.setattr(
            adapter,
            "_start_session_processing",
            lambda event, session_key: started.append(session_key),
        )

        event = MessageEvent(text="/stop", source=source, message_id="2")
        await adapter.handle_message(event)

        assert event.source.session_binding == binding
        assert dispatched == [(bound_session_key, "stop")]
        assert started == []

    @pytest.mark.asyncio
    async def test_unbound_event_resolves_session_binding_once(self, monkeypatch):
        adapter = StubAdapter()
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))
        event = MessageEvent(
            text="hello",
            source=_source("-1001234", "group", thread_id="75911"),
            message_id="2",
        )
        hook_calls = []

        def invoke_hook(name, **kwargs):
            hook_calls.append(name)
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
        monkeypatch.setattr(adapter, "_start_session_processing", lambda event, session_key: None)

        adapter._apply_session_binding(event)
        await adapter.handle_message(event)

        assert hook_calls == ["pre_session_binding"]

    @pytest.mark.asyncio
    async def test_photo_followup_is_queued_without_interrupt(self):
        """Photo follow-ups should queue behind the active run instead of interrupting it."""
        adapter = StubAdapter()
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        source = _source("-1001234", "group")
        session_key = build_session_key(source)
        interrupt_event = asyncio.Event()
        adapter._active_sessions[session_key] = interrupt_event

        event = MessageEvent(
            text="caption",
            source=source,
            message_type=MessageType.PHOTO,
            message_id="2",
            media_urls=["/tmp/photo-a.jpg"],
            media_types=["image/jpeg"],
        )
        await adapter.handle_message(event)

        queued = adapter._pending_messages[session_key]
        assert queued is event
        assert queued.media_urls == ["/tmp/photo-a.jpg"]
        assert interrupt_event.is_set() is False
