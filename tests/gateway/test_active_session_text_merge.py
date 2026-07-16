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
from dataclasses import replace
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
    merge_pending_message_event,
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
    reply_to_message_id: str | None = None,
    reply_to_text: str | None = None,
    reply_to_author_id: str | None = None,
    reply_to_author_name: str | None = None,
    reply_to_is_own_message: bool = False,
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
        reply_to_message_id=reply_to_message_id,
        reply_to_text=reply_to_text,
        reply_to_author_id=reply_to_author_id,
        reply_to_author_name=reply_to_author_name,
        reply_to_is_own_message=reply_to_is_own_message,
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


@pytest.mark.parametrize("media_urls,media_types", [
    ([], []),
    (["/tmp/q.png"], ["image/png"]),
])
@pytest.mark.parametrize("authorized", [None, False, True])
def test_pending_message_merge_keeps_incoming_reply_context(media_urls, media_types, authorized):
    existing = _make_event("one")
    incoming = _make_event("two")
    incoming.media_urls, incoming.media_types = list(media_urls), list(media_types)
    incoming.reply_to_message_id, incoming.reply_to_text = "$photo", "[image]"
    incoming.reply_to_author_id, incoming.reply_to_author_name = "@alice:example.org", "Alice"
    incoming.reply_to_author_authorized = authorized
    expected = replace(
        existing, text="one\n\ntwo" if media_urls else "one\ntwo",
        media_urls=list(media_urls), media_types=list(media_types),
        media_text_inlined=[None] * len(media_urls),
        reply_to_message_id="$photo", reply_to_text="[image]",
        reply_to_author_id="@alice:example.org", reply_to_author_name="Alice",
        reply_to_author_authorized=authorized, merged_message_ids=[incoming.message_id],
    )
    pending = {"session": existing}

    merge_pending_message_event(pending, "session", incoming, merge_text=True)

    assert pending == {"session": expected}


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
@pytest.mark.parametrize(
    "mode,reply_ids,attachments,expected_turns",
    [
        pytest.param(
            "pending",
            (None, "a"),
            (None, None),
            (("one\ntwo", 1, 0, (0, 1)),),
            id="pending-incoming-quote",
        ),
        pytest.param(
            "pending",
            ("a", None),
            (None, None),
            (("one\ntwo", 0, 0, (0, 1)),),
            id="pending-keeps-quote",
        ),
        pytest.param(
            "pending",
            ("a", "b"),
            (None, None),
            (("one\ntwo", 0, 0, (0, 1)),),
            id="pending-single-slot-fallback",
        ),
        pytest.param(
            "pending",
            (None, "a"),
            (None, "image"),
            (("one\n\ntwo", 1, 0, (0, 1)),),
            id="pending-quoted-image",
        ),
        pytest.param(
            "debounce",
            (None, "a"),
            (None, None),
            (("one\ntwo", 1, 1, (0, 1)),),
            id="debounce-incoming-quote",
        ),
        pytest.param(
            "debounce",
            ("a", "a"),
            (None, None),
            (("one\ntwo", 0, 1, (0, 1)),),
            id="debounce-same-quote",
        ),
        pytest.param(
            "debounce",
            ("a", "b"),
            (None, None),
            (("one", 0, 0, (0,)), ("two", 1, 1, (1,))),
            id="debounce-distinct-quotes",
        ),
        pytest.param(
            "debounce",
            ("a", "b", "c"),
            (None, None, None),
            (("one", 0, 0, (0,)), ("two\nthree", 1, 2, (1, 2))),
            id="debounce-runnerless-fallback",
        ),
        pytest.param(
            "debounce",
            (None, "a"),
            (None, "image"),
            (("one\ntwo", 1, 1, (0, 1)),),
            id="debounce-quoted-image",
        ),
        pytest.param(
            "debounce",
            (None, "a"),
            (None, "document"),
            (("one\ntwo", 1, 1, (0, 1)),),
            id="debounce-not-inlined-document",
        ),
        pytest.param(
            "debounce",
            (None, "a"),
            ("legacy-image", "document"),
            (("one\ntwo", 1, 1, (0, 1)),),
            id="debounce-pads-legacy-inline-flags",
        ),
    ],
)
async def test_busy_merges_preserve_reply_context_and_attachments(
    mode, reply_ids, attachments, expected_turns
):
    events = []
    for word, reply_id, attachment in zip(
        ("one", "two", "three"), reply_ids, attachments
    ):
        event = _make_event(
            word,
            reply_to_message_id=reply_id,
            reply_to_text=f"quote {reply_id}" if reply_id else None,
            reply_to_author_id=f"author-{reply_id}" if reply_id else None,
            reply_to_author_name=f"Author {reply_id}" if reply_id else None,
            reply_to_is_own_message=bool(reply_id),
        )
        if attachment:
            is_document = attachment == "document"
            event.media_urls = [
                f"/tmp/{word}.txt" if is_document else f"/tmp/{word}.png"
            ]
            event.media_types = ["text/plain" if is_document else "image/png"]
            event.media_text_inlined = [] if attachment == "legacy-image" else [False]
        events.append(event)

    expected = []
    for text, quote_index, anchor_index, members in expected_turns:
        quote = events[quote_index]
        expected.append(
            replace(
                events[members[0]],
                text=text,
                merged_message_ids=[events[index].message_id for index in members if index != anchor_index],
                message_id=events[anchor_index].message_id,
                reply_to_message_id=quote.reply_to_message_id,
                reply_to_text=quote.reply_to_text,
                reply_to_author_id=quote.reply_to_author_id,
                reply_to_author_name=quote.reply_to_author_name,
                reply_to_is_own_message=quote.reply_to_is_own_message,
                media_urls=[
                    path for index in members for path in events[index].media_urls
                ],
                media_types=[
                    kind for index in members for kind in events[index].media_types
                ],
                media_text_inlined=[
                    flag
                    for index in members
                    for flag in (
                        events[index].media_text_inlined
                        or [None] * len(events[index].media_urls)
                    )
                ],
            )
        )

    if mode == "pending":
        pending = {"session": events[0]}
        for event in events[1:]:
            merge_pending_message_event(pending, "session", event, merge_text=True)
        assert list(pending.values()) == expected
        return

    adapter = _make_adapter()
    session_key = build_session_key(events[0].source)
    adapter._active_sessions[session_key] = asyncio.Event()
    for event in events:
        await adapter.handle_message(event)
    await adapter._flush_text_debounce_now(session_key)
    actual = [adapter._pending_messages[session_key]]
    buffered = adapter._text_debounce.get(session_key)
    if buffered is not None:
        actual.append(buffered.event)
    adapter._discard_text_debounce(session_key)
    assert actual == expected


@pytest.mark.asyncio
async def test_queue_debounce_preserves_same_reply_context():
    adapter = _make_adapter()
    first = _make_event(
        "one",
        reply_to_message_id="reply-1",
        reply_to_text="quoted",
        reply_to_author_id="author-1",
        reply_to_author_name="Author One",
        reply_to_is_own_message=True,
    )
    session_key = build_session_key(first.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(first)
    await adapter.handle_message(
        _make_event(
            "two",
            reply_to_message_id="reply-1",
            reply_to_text="quoted",
            reply_to_author_id="author-1",
            reply_to_author_name="Author One",
            reply_to_is_own_message=True,
        )
    )

    merged = _debounced_event(adapter, session_key)
    assert merged.text == "one\ntwo"
    assert merged.message_id == "msg-two"
    assert (
        merged.reply_to_message_id,
        merged.reply_to_text,
        merged.reply_to_author_id,
        merged.reply_to_author_name,
        merged.reply_to_is_own_message,
    ) == ("reply-1", "quoted", "author-1", "Author One", True)
    adapter._discard_text_debounce(session_key)


@pytest.mark.asyncio
async def test_queue_debounce_splits_incompatible_reply_contexts():
    adapter = _make_adapter()
    first = _make_event(
        "one",
        reply_to_message_id="reply-1",
        reply_to_text="first quote",
        reply_to_author_id="author-1",
        reply_to_author_name="Author One",
    )
    session_key = build_session_key(first.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(first)
    await adapter.handle_message(
        _make_event(
            "two",
            reply_to_message_id="reply-2",
            reply_to_text="second quote",
            reply_to_author_id="author-2",
            reply_to_author_name="Author Two",
        )
    )

    pending = adapter._pending_messages[session_key]
    queued = _debounced_event(adapter, session_key)
    assert pending.text == "one"
    assert (
        pending.reply_to_message_id,
        pending.reply_to_text,
        pending.reply_to_author_id,
        pending.reply_to_author_name,
    ) == ("reply-1", "first quote", "author-1", "Author One")
    assert queued.text == "two"
    assert (
        queued.reply_to_message_id,
        queued.reply_to_text,
        queued.reply_to_author_id,
        queued.reply_to_author_name,
    ) == ("reply-2", "second quote", "author-2", "Author Two")
    adapter._discard_text_debounce(session_key)
