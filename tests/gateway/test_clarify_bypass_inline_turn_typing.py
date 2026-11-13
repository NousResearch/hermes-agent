"""Typing indicator for a clarify bypass that starts a fresh turn (#154524).

A plugin command's ``ask_choice`` card holds the chat with no agent turn behind it. Prose that
cancels the card becomes the chat's next turn, which ``_handle_message_while_active`` dispatches
inline — a path with no session lifecycle and, before this fix, no typing refresh, so the reply
arrived with no indicator for the whole turn.

The clarify *tool*'s own card is the opposite case: its turn is already running (blocked inside the
tool), so that turn's refresh must stay the only one for the chat.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _BypassAdapter(BasePlatformAdapter):
    def __init__(self, typing_indicator: bool = True):
        super().__init__(
            PlatformConfig(enabled=True, token="test", typing_indicator=typing_indicator),
            Platform.TELEGRAM,
        )
        self.send_typing = AsyncMock(return_value=None)
        self.stop_typing = AsyncMock(return_value=None)
        self.handler_calls = 0

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _event(text="count from 1 to 10"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="private",
                             user_id="user1"),
        message_id="msg1",
    )


def _session_key(adapter, event):
    return build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )


@pytest.fixture(autouse=True)
def _clear_clarify_state():
    from tools import clarify_gateway as cm

    def _clear():
        with cm._lock:
            cm._entries.clear()
            cm._session_index.clear()
            cm._notify_cbs.clear()

    _clear()
    yield
    _clear()


def _register_choice_card(session_key: str) -> None:
    """The pending card a plugin command's ``ctx.ask_choice`` leaves behind."""
    from tools import clarify_gateway as cm

    cm.register("clarify-1", session_key, "Pick one", ["A", "B"])


def _slow_handler(adapter: _BypassAdapter):
    """A handler that stands in for the turn the prose starts: it yields so the spawned typing
    refresh can run its first tick, then returns the reply."""
    async def _handle(_event):
        adapter.handler_calls += 1
        for _ in range(10):
            await asyncio.sleep(0)
        return "1, 2, 3"

    return _handle


def _runner(*, running: bool):
    return SimpleNamespace(_is_session_running=lambda _key: running)


@pytest.mark.asyncio
async def test_plugin_card_prose_shows_typing_for_the_turn_it_starts():
    adapter = _BypassAdapter()
    adapter._message_handler = _slow_handler(adapter)
    adapter.gateway_runner = _runner(running=False)
    event = _event()
    session_key = _session_key(adapter, event)
    adapter._active_sessions[session_key] = asyncio.Event()
    _register_choice_card(session_key)

    await adapter._handle_message_while_active(event, session_key)

    assert adapter.handler_calls == 1
    assert adapter.send_typing.await_count >= 1, "the turn started by the prose got no typing indicator"
    # The bracket must clear the indicator again instead of leaving a chat paused.
    assert adapter.stop_typing.await_count >= 1
    assert session_key not in adapter._typing_paused and (
        event.source.chat_id, None) not in adapter._typing_paused


@pytest.mark.asyncio
async def test_plugin_choice_reply_does_not_fake_a_new_turn_typing_indicator():
    """A typed choice settles an idle plugin card; it does not launch a turn."""
    adapter = _BypassAdapter()
    adapter.gateway_runner = _runner(running=False)
    event = _event("1")
    session_key = _session_key(adapter, event)
    adapter._active_sessions[session_key] = asyncio.Event()
    _register_choice_card(session_key)

    async def answer_card(inbound):
        from tools import clarify_gateway as cm
        adapter.handler_calls += 1
        assert cm.attempt_text_response_for_session(session_key, inbound.text) == cm.TEXT_RESOLVED
        await asyncio.sleep(0)  # the real gateway can await card retirement
        return ""

    adapter._message_handler = answer_card
    await adapter._handle_message_while_active(event, session_key)

    assert adapter.handler_calls == 1
    assert adapter.send_typing.await_count == 0


@pytest.mark.asyncio
async def test_answer_to_a_running_turns_clarify_keeps_that_turns_own_indicator():
    adapter = _BypassAdapter()
    adapter._message_handler = _slow_handler(adapter)
    adapter.gateway_runner = _runner(running=True)
    event = _event("A")
    session_key = _session_key(adapter, event)
    adapter._active_sessions[session_key] = asyncio.Event()
    _register_choice_card(session_key)

    await adapter._handle_message_while_active(event, session_key)

    assert adapter.handler_calls == 1
    assert adapter.send_typing.await_count == 0


@pytest.mark.asyncio
async def test_typing_indicator_off_keeps_the_bypass_silent():
    adapter = _BypassAdapter(typing_indicator=False)
    adapter._message_handler = _slow_handler(adapter)
    adapter.gateway_runner = _runner(running=False)
    event = _event()
    session_key = _session_key(adapter, event)
    adapter._active_sessions[session_key] = asyncio.Event()
    _register_choice_card(session_key)

    await adapter._handle_message_while_active(event, session_key)

    assert adapter.handler_calls == 1
    assert adapter.send_typing.await_count == 0


@pytest.mark.asyncio
async def test_adapter_without_a_runner_keeps_the_previous_behaviour():
    """A standalone adapter (tests, relays) cannot tell a plugin card from a clarify answer: it
    must not guess, and the reply still goes out."""
    adapter = _BypassAdapter()
    adapter._message_handler = _slow_handler(adapter)
    event = _event()
    session_key = _session_key(adapter, event)
    adapter._active_sessions[session_key] = asyncio.Event()
    _register_choice_card(session_key)

    await adapter._handle_message_while_active(event, session_key)

    assert adapter.handler_calls == 1
    assert adapter.send_typing.await_count == 0
