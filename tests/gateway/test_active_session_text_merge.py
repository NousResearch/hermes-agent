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
    SendResult,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


def _make_event(
    text: str,
    chat_id: str = "12345",
    *,
    chat_type: str = "dm",
    user_id: str = "u1",
    user_name: str | None = None,
    thread_id: str | None = None,
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
    )


class _DummyAdapter(BasePlatformAdapter):  # type: ignore[misc]
    async def connect(self, *, is_reconnect: bool = False):
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
async def test_non_dm_message_does_not_wait_for_topic_recovery_executor(monkeypatch):
    """Group messages must not queue behind the shared thread pool.

    Topic recovery only applies to Telegram DM topic mode. Offloading that
    no-op check for every group message makes ingress wait behind unrelated
    blocking jobs when the default executor is saturated.
    """
    adapter = _make_adapter()
    recovery = MagicMock(return_value=None)
    adapter.set_topic_recovery_fn(recovery)
    executor_called = False
    never_release = asyncio.Event()

    async def _blocked_to_thread(*args, **kwargs):
        nonlocal executor_called
        executor_called = True
        await never_release.wait()

    monkeypatch.setattr(asyncio, "to_thread", _blocked_to_thread)

    await asyncio.wait_for(
        adapter.handle_message(_make_event("/status", chat_type="group")),
        timeout=1.0,
    )
    await asyncio.sleep(0)

    assert executor_called is False
    recovery.assert_not_called()


@pytest.mark.asyncio
async def test_dm_topic_recovery_stays_offloaded(monkeypatch):
    """Real Telegram DM topic recovery must still run outside the event loop."""
    adapter = _make_adapter()
    recovery = MagicMock(return_value="topic-222")
    adapter.set_topic_recovery_fn(recovery)
    offloaded = False

    async def _inline_to_thread(func, *args, **kwargs):
        nonlocal offloaded
        offloaded = True
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread)
    event = _make_event("hello", chat_type="dm", thread_id="1")
    original_source = event.source

    await adapter.handle_message(event)
    await asyncio.sleep(0)

    assert offloaded is True
    assert recovery.call_count == 1
    assert recovery.call_args.args[0] is original_source
    assert event.source.thread_id == "topic-222"


@pytest.mark.asyncio
async def test_lobby_reply_keys_busy_path_on_recovered_topic(tmp_path):
    """Adapter-side topic recovery must agree with the turn's session key.

    Uses the runner's REAL ``_recover_telegram_topic_thread_id`` as the hook (a MagicMock
    cannot catch a missing ``is_reply=`` pass-through): a stripped lobby REPLY arriving while
    its bound topic runs must be keyed on THAT topic's session by the adapter's busy guard —
    not start a second, parallel turn on the lobby lane (the in-turn recovery in run_turn
    would pin the same topic, splitting the guard key from the turn's session key).
    """
    from gateway.run import GatewayRunner
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="12345", user_id="u1")
    db.create_session(session_id="sess-topic", source="telegram", user_id="u1")
    topic_source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm",
        user_id="u1", thread_id="222",
    )
    db.bind_telegram_topic(
        chat_id="12345", thread_id="222", user_id="u1",
        session_key=build_session_key(topic_source), session_id="sess-topic",
    )

    runner = object.__new__(GatewayRunner)
    runner._session_db = db

    adapter = _make_adapter()
    adapter._busy_text_mode = ""  # direct merge, no debounce
    adapter.set_topic_recovery_fn(runner._recover_telegram_topic_thread_id)

    lobby_event = _make_event("hello", thread_id=None)
    lobby_key = build_session_key(lobby_event.source)
    topic_key = build_session_key(topic_source)
    assert lobby_key != topic_key
    adapter._active_sessions[topic_key] = asyncio.Event()  # a turn is running in topic 222

    # A stripped lobby REPLY (Telegram DM topic mode drops message_thread_id on replies):
    # recovery pins it to topic 222 BEFORE session keying, so it queues onto the topic's
    # running turn and never touches the lobby lane.
    reply = _make_event("follow up", thread_id=None)
    reply.reply_to_message_id = "41"
    await adapter.handle_message(reply)

    assert str(reply.source.thread_id) == "222"
    assert adapter._pending_messages[topic_key].text == "follow up"
    assert lobby_key not in adapter._pending_messages

    # A FRESH lobby message (not a reply) stays unrouted at this layer: the fresh-lobby
    # router owns it. Recovery returns None, nothing is queued onto the busy topic.
    fresh = _make_event("brand new topic", thread_id=None)
    await adapter.handle_message(fresh)

    assert fresh.source.thread_id is None
    assert adapter._pending_messages[topic_key].text == "follow up"  # unchanged
    db.close()


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
