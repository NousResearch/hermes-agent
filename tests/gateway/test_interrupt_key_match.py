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
from gateway.session import SessionSource, build_session_key


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


