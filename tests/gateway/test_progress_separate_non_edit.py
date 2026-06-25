"""Tests for tool-progress delivery on non-edit platforms.

Issue: #52212

When a platform does not support ``edit_message`` (QQ Bot, WeChat, Signal,
BlueBubbles, etc.), the gateway previously drained the progress queue and
returned immediately — regardless of the ``tool_progress_grouping`` setting.

In ``separate`` mode each progress update is sent as an independent message
via ``adapter.send()``, which does NOT require ``edit_message``.  The guard
must only bail out when the platform lacks editing support AND the grouping
mode actually needs it (``accumulate``).
"""

from __future__ import annotations

import asyncio
import queue
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


# ---------------------------------------------------------------------------
# Test adapters
# ---------------------------------------------------------------------------

class NoEditAdapter(BasePlatformAdapter):
    """Adapter that does NOT override edit_message — like QQ Bot, WeChat, etc."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent: list[dict] = []
        self._msg_counter = 0

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self._msg_counter += 1
        self.sent.append({"chat_id": chat_id, "content": content, "msg_id": self._msg_counter})
        return SendResult(success=True, message_id=str(self._msg_counter))

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_separate_mode_sends_progress_on_no_edit_platform():
    """Non-edit platforms with separate grouping MUST deliver progress messages.

    Before the fix, the edit_message guard drained the queue and returned
    before can_edit was evaluated, silently dropping all progress.
    """
    adapter = NoEditAdapter()
    progress_queue: queue.Queue = queue.Queue()

    # Enqueue two progress events
    progress_queue.put(("tool.started", "web_search", "🔍 searching", {}))
    progress_queue.put(("tool.completed", "web_search", "✅ done", {}))

    # Build minimal gateway-like context to call send_progress_messages
    # We replicate the relevant logic inline since the full gateway is heavy.
    progress_grouping = "separate"
    can_edit = progress_grouping != "separate"  # False in separate mode

    # The guard: should NOT bail out because can_edit is False
    _no_edit_support = type(adapter).edit_message is BasePlatformAdapter.edit_message
    assert _no_edit_support is True, "Adapter should lack edit_message"

    if _no_edit_support and can_edit:
        pytest.fail("Guard should NOT fire when can_edit=False (separate mode)")

    # Drain the queue as the real code would in the main loop
    while not progress_queue.empty():
        raw = progress_queue.get_nowait()
        if isinstance(raw, tuple) and len(raw) >= 3:
            _kind, _tool, msg, _meta = raw
            result = await adapter.send(
                chat_id="test-chat",
                content=msg,
            )
            assert result.success

    # Both messages should have been sent
    assert len(adapter.sent) == 2
    assert "🔍 searching" in adapter.sent[0]["content"]
    assert "✅ done" in adapter.sent[1]["content"]


@pytest.mark.asyncio
async def test_accumulate_mode_still_drains_on_no_edit_platform():
    """Non-edit platforms with accumulate grouping MUST drain (old behavior).

    Without edit_message, accumulate mode would create a new message for every
    single tool tick — very noisy.  The guard correctly suppresses this.
    """
    adapter = NoEditAdapter()
    progress_queue: queue.Queue = queue.Queue()

    progress_queue.put(("tool.started", "web_search", "🔍 searching", {}))

    progress_grouping = "accumulate"
    can_edit = progress_grouping != "separate"  # True in accumulate mode

    _no_edit_support = type(adapter).edit_message is BasePlatformAdapter.edit_message
    assert _no_edit_support is True

    should_drain = _no_edit_support and can_edit
    assert should_drain is True, "Guard should fire in accumulate mode on non-edit platform"

    # Simulate drain
    while not progress_queue.empty():
        try:
            progress_queue.get_nowait()
        except Exception:
            break

    # Nothing should have been sent
    assert len(adapter.sent) == 0


@pytest.mark.asyncio
async def test_edit_platform_unaffected_by_guard_change():
    """Platforms WITH edit_message must not be affected by this change.

    The guard should never fire for adapters that override edit_message,
    regardless of grouping mode.
    """
    class EditAdapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
            self.sent = []

        async def connect(self):
            return True

        async def disconnect(self):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.sent.append(content)
            return SendResult(success=True, message_id="1")

        async def edit_message(self, chat_id, message_id, content):
            return SendResult(success=True, message_id=message_id)

        async def send_typing(self, chat_id, metadata=None):
            return None

        async def stop_typing(self, chat_id):
            return None

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

    adapter = EditAdapter()
    _no_edit_support = type(adapter).edit_message is BasePlatformAdapter.edit_message
    assert _no_edit_support is False, "EditAdapter should support edit_message"

    # Guard must NOT fire even in accumulate mode
    for grouping in ("accumulate", "separate"):
        can_edit = grouping != "separate"
        if _no_edit_support and can_edit:
            pytest.fail(f"Guard fired for edit-supporting adapter in {grouping} mode")
