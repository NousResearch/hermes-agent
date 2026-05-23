"""Regression tests for active-session TEXT follow-up queueing.

When the agent is actively running, rapid text follow-ups should survive as
one next-turn pending message instead of clobbering each other. In
``busy_text_mode=queue`` those active follow-ups first pass through a short
debounce so bursty multi-message thoughts are merged before the active drain
hands off the next turn.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Minimal telegram stub so importing gateway.platforms.base does not pull
# in the real python-telegram-bot dependency.
_tg = sys.modules.get("telegram") or types.ModuleType("telegram")
_tg.constants = sys.modules.get("telegram.constants") or types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.PRIVATE = "private"
_ct.GROUP = "group"
_ct.SUPERGROUP = "supergroup"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


def _make_event(
    text: str,
    chat_id: str = "12345",
    *,
    chat_type: str = "dm",
    user_id: str = "u1",
    user_name: str | None = None,
    thread_id: str | None = None,
    internal: bool = False,
) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name=user_name,
        thread_id=thread_id,
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=f"msg-{text[:8]}",
        internal=internal,
    )


class _DummyAdapter(BasePlatformAdapter):  # type: ignore[misc]
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, *args, **kwargs):
        return SendResult(success=True, message_id="x")


def _make_initialized_adapter() -> BasePlatformAdapter:
    return _DummyAdapter(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)


def _make_adapter() -> BasePlatformAdapter:
    """Build a BasePlatformAdapter without running its heavy __init__."""
    adapter = object.__new__(_DummyAdapter)
    adapter.config = PlatformConfig(enabled=True, token="***")
    adapter.platform = Platform.TELEGRAM
    adapter._message_handler = AsyncMock(return_value=None)
    adapter._busy_session_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._session_tasks = {}
    adapter._background_tasks = set()
    adapter._post_delivery_callbacks = {}
    adapter._expected_cancelled_tasks = set()
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._running = True
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 0.1
    adapter._busy_text_hard_cap_seconds = 1.0
    adapter._text_debounce = {}
    adapter._auto_tts_default = False
    adapter._auto_tts_enabled_chats = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._typing_paused = set()
    return adapter


def _debounced_event(adapter: BasePlatformAdapter, session_key: str) -> MessageEvent:
    return adapter._text_debounce[session_key].event


@pytest.mark.asyncio
async def test_rapid_text_followups_accumulate_instead_of_replacing():
    """Rapid TEXT follow-ups must all survive in the pending event."""
    adapter = _make_adapter()
    adapter._busy_text_mode = ""  # direct-merge behavior, no debounce
    first = _make_event("part one")
    session_key = build_session_key(first.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(_make_event("part two"))
    await adapter.handle_message(_make_event("part three"))

    pending = adapter._pending_messages[session_key]
    assert pending.text == "part two\npart three"
    assert not adapter._active_sessions[session_key].is_set()


@pytest.mark.asyncio
async def test_debounce_buffers_rapid_text_then_flushes_to_pending():
    adapter = _make_adapter()
    adapter._busy_text_debounce_seconds = 0.05

    first = _make_event("part one")
    session_key = build_session_key(first.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(_make_event("part two"))
    assert session_key in adapter._text_debounce
    assert _debounced_event(adapter, session_key).text == "part two"
    assert session_key not in adapter._pending_messages

    await adapter.handle_message(_make_event("part three"))
    assert _debounced_event(adapter, session_key).text == "part two\npart three"

    await asyncio.sleep(0.15)

    assert session_key not in adapter._text_debounce
    assert adapter._pending_messages[session_key].text == "part two\npart three"


@pytest.mark.asyncio
async def test_debounce_resets_timer_on_new_arrival():
    adapter = _make_adapter()
    adapter._busy_text_debounce_seconds = 0.1

    first = _make_event("one")
    session_key = build_session_key(first.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(first)
    task1 = adapter._text_debounce[session_key].task
    assert task1 is not None
    assert not task1.done()

    await adapter.handle_message(_make_event("two"))
    task2 = adapter._text_debounce[session_key].task
    assert task2 is not None
    assert task2 is not task1
    await asyncio.sleep(0)
    assert task1.cancelled() or task1.done()
    assert adapter._text_debounce[session_key].task is task2

    await adapter.handle_message(_make_event("three"))
    task3 = adapter._text_debounce[session_key].task
    assert task3 is not None
    assert task3 is not task2

    await asyncio.sleep(0.2)
    assert session_key not in adapter._text_debounce
    assert adapter._pending_messages[session_key].text == "one\ntwo\nthree"


@pytest.mark.asyncio
async def test_active_drain_force_flushes_debounce_before_release():
    adapter = _make_adapter()
    adapter._busy_text_debounce_seconds = 1.0
    processed: list[str] = []

    async def _handler(event):
        processed.append(event.text)
        if event.text == "current":
            await adapter.handle_message(_make_event("follow up"))
        return None

    adapter._message_handler = _handler
    current = _make_event("current")
    session_key = build_session_key(current.source)

    task = asyncio.create_task(adapter._process_message_background(current, session_key))
    adapter._session_tasks[session_key] = task
    await asyncio.wait_for(task, timeout=1.0)

    for _ in range(20):
        if processed == ["current", "follow up"] and session_key not in adapter._active_sessions:
            break
        await asyncio.sleep(0.05)

    assert processed == ["current", "follow up"]
    assert session_key not in adapter._text_debounce
    assert session_key not in adapter._pending_messages
    assert session_key not in adapter._active_sessions


@pytest.mark.asyncio
async def test_force_flush_cancels_timer_without_duplicate_processing():
    adapter = _make_adapter()
    adapter._busy_text_debounce_seconds = 0.2

    event = _make_event("queued once")
    session_key = build_session_key(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)
    timer_task = adapter._text_debounce[session_key].task

    flushed = await adapter._flush_text_debounce_now(session_key)
    assert flushed is True
    assert session_key not in adapter._text_debounce
    assert adapter._pending_messages[session_key].text == "queued once"

    await asyncio.sleep(0.3)
    assert timer_task is not None
    assert timer_task.cancelled() or timer_task.done()
    assert adapter._pending_messages[session_key].text == "queued once"


@pytest.mark.asyncio
async def test_text_debounce_does_not_merge_different_senders():
    adapter = _make_adapter()
    adapter._busy_text_debounce_seconds = 1.0

    first = _make_event(
        "from alice",
        chat_type="group",
        user_id="alice",
        user_name="Alice",
        thread_id="topic-1",
    )
    second = _make_event(
        "from bob",
        chat_type="group",
        user_id="bob",
        user_name="Bob",
        thread_id="topic-1",
    )
    session_key = build_session_key(first.source)
    assert session_key == build_session_key(second.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(first)
    await adapter.handle_message(second)

    assert adapter._pending_messages[session_key].text == "from alice"
    assert _debounced_event(adapter, session_key).text == "from bob"


@pytest.mark.asyncio
async def test_control_and_clarify_messages_bypass_text_debounce():
    adapter = _make_adapter()
    started: list[str] = []

    def _fake_start(event, session_key, *, interrupt_event=None):
        started.append(event.text)
        return True

    adapter._start_session_processing = _fake_start  # type: ignore[method-assign]

    await adapter.handle_message(_make_event("/status"))
    assert started == ["/status"]
    assert adapter._text_debounce == {}

    answer = _make_event("clarify answer")
    session_key = build_session_key(answer.source)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._message_handler = AsyncMock(return_value=None)

    with patch("tools.clarify_gateway.get_pending_for_session", return_value=object()):
        await adapter.handle_message(answer)

    adapter._message_handler.assert_awaited_once_with(answer)
    assert session_key not in adapter._text_debounce
    assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_debounce_skipped_when_busy_text_mode_not_queue():
    adapter = _make_adapter()
    adapter._busy_text_mode = ""
    event = _make_event("direct merge")
    session_key = build_session_key(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    assert adapter._pending_messages[session_key].text == "direct merge"
    assert session_key not in adapter._text_debounce


def test_debounce_respects_env_var_override(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_TEXT_DEBOUNCE_SECONDS", "2.5")
    adapter = _make_initialized_adapter()
    assert adapter._busy_text_debounce_seconds == 2.5


@pytest.mark.asyncio
async def test_debounce_cleanup_in_cancel_background_tasks():
    adapter = _make_adapter()
    adapter._busy_text_debounce_seconds = 1.0

    event = _make_event("cleanup test")
    session_key = build_session_key(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter.handle_message(event)

    assert session_key in adapter._text_debounce

    await adapter.cancel_background_tasks()

    assert session_key not in adapter._text_debounce


@pytest.mark.asyncio
async def test_single_followup_is_stored_as_is():
    adapter = _make_adapter()
    adapter._busy_text_mode = ""
    first = _make_event("only one")
    session_key = build_session_key(first.source)

    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter.handle_message(first)

    pending = adapter._pending_messages[session_key]
    assert pending is first
    assert pending.text == "only one"
    assert not adapter._active_sessions[session_key].is_set()


def test_adapter_defaults_to_queue_mode(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_TEXT_MODE", raising=False)
    adapter = _make_initialized_adapter()
    assert adapter._busy_text_mode == "queue"
    assert adapter._is_queue_text_debounce_candidate(_make_event("hello"))


def test_adapter_is_queue_text_debounce_candidate_by_default():
    adapter = _make_adapter()
    assert adapter._is_queue_text_debounce_candidate(_make_event("hello world"))


def test_command_messages_bypass_debounce_even_in_queue_mode():
    adapter = _make_adapter()
    assert not adapter._is_queue_text_debounce_candidate(_make_event(""))
    assert not adapter._is_queue_text_debounce_candidate(_make_event("/stop"))


def test_busy_text_mode_respects_env_var_override(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_TEXT_MODE", "interrupt")
    adapter = _make_initialized_adapter()
    assert adapter._busy_text_mode == "interrupt"
    assert not adapter._is_queue_text_debounce_candidate(_make_event("test"))


@pytest.mark.asyncio
async def test_internal_followup_queues_without_interrupting_active_session():
    """Synthetic process/watch events must not masquerade as live user input.

    Background process notifications use ``MessageEvent.internal=True`` and may
    arrive while the session they belong to is already processing a real user
    turn. They should be queued for the next turn without setting the interrupt
    guard; otherwise the gateway emits a bogus "Interrupting current task" ack
    and aborts user work even though no user sent a new message.
    """
    adapter = _make_adapter()
    first = _make_event("real user work")
    session_key = build_session_key(first.source)

    guard = asyncio.Event()
    adapter._active_sessions[session_key] = guard
    internal_event = _make_event(
        '[IMPORTANT: Background process proc_x matched watch pattern "ready".]',
        internal=True,
    )

    await adapter.handle_message(internal_event)

    pending = adapter._pending_messages[session_key]
    assert pending is internal_event
    assert pending.internal is True
    assert not guard.is_set(), "internal events must not interrupt the active user turn"
