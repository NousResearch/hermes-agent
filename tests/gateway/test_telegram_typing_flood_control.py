"""Issue #41168 reproduction specs for Telegram typing + flood control.

The user report is not "typing API itself got stuck forever". The stronger
source-level explanation is:

1. ``notify_on_complete=True`` injects a synthetic background-completion turn.
2. ``BasePlatformAdapter._process_message_background()`` starts
   ``_keep_typing(...)`` immediately for that turn.
3. Final response delivery then goes through ``_send_with_retry()``, which
   calls ``TelegramAdapter.send()``.
4. ``TelegramAdapter.send()`` handles Telegram ``retry_after`` (flood control)
   *inside* the send coroutine and can therefore stay pending for a long time.
5. The typing cleanup lives in the outer ``finally`` block, so it does not run
   until the send coroutine returns.

These tests preserve that pending-send reproduction and verify that the typing
loop pauses while the outer cleanup is still waiting for delivery to finish.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
)
from gateway.session import SessionSource, build_session_key


class _RetryAfter(Exception):
    def __init__(self, seconds: float):
        super().__init__(f"Retry after {seconds}")
        self.retry_after = seconds


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    return TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))


def _make_internal_event(chat_id: str = "123") -> MessageEvent:
    return MessageEvent(
        text="[IMPORTANT: Background process completed]",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="dm",
        ),
        internal=True,
        message_id="999",
    )


@pytest.mark.asyncio
async def test_telegram_send_can_stay_pending_after_last_retry_after_log():
    """Reproduce the "last flood-control log, then silence" shape from #41168.

    ``TelegramAdapter.send()`` handles ``retry_after`` internally. Once it has
    entered the final send attempt, the coroutine can remain pending with no
    further flood-control log lines until that attempt returns.
    """
    adapter = _make_adapter()

    attempts = {"count": 0}
    final_attempt_started = asyncio.Event()
    release_final_attempt = asyncio.Event()

    async def mock_send_message(**kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _RetryAfter(0.01)
        final_attempt_started.set()
        await release_final_attempt.wait()
        return SimpleNamespace(message_id=500)

    adapter._bot = SimpleNamespace(send_message=mock_send_message, send_chat_action=None)

    send_task = asyncio.create_task(adapter.send(chat_id="123", content="done"))
    try:
        await asyncio.wait_for(final_attempt_started.wait(), timeout=1.0)
        await asyncio.sleep(0.05)

        assert attempts["count"] == 3, (
            "send() should already be inside its final Telegram send attempt "
            "after two retry_after backoffs"
        )
        assert not send_task.done(), (
            "send() unexpectedly finished; the reproduction needs the final "
            "Telegram send attempt to remain pending after the last retry_after log"
        )

        release_final_attempt.set()
        result = await asyncio.wait_for(send_task, timeout=1.0)

        assert result.success is True
        assert result.message_id == "500"
    finally:
        release_final_attempt.set()
        if not send_task.done():
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_process_message_background_pauses_typing_during_retry_after_send():
    """Fix spec for #41168.

    Once Telegram send delivery enters flood-control retry handling, the
    background typing loop should pause instead of refreshing indefinitely while
    the final send attempt is still pending.
    """
    adapter = _make_adapter()
    event = _make_internal_event()
    session_key = build_session_key(event.source)

    typing_calls: list[str] = []
    paused_chat_ids: list[str] = []
    stop_typing_called = asyncio.Event()
    final_attempt_started = asyncio.Event()
    release_final_attempt = asyncio.Event()
    attempts = {"count": 0}
    pause_seen = asyncio.Event()

    async def mock_send_message(**kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _RetryAfter(0.01)
        final_attempt_started.set()
        await release_final_attempt.wait()
        return SimpleNamespace(message_id=500)

    async def mock_send_typing(**kwargs):
        typing_calls.append(str(kwargs["chat_id"]))

    async def mock_stop_typing(chat_id: str):
        stop_typing_called.set()

    original_pause = adapter.pause_typing_for_chat

    def recording_pause(chat_id: str) -> None:
        paused_chat_ids.append(str(chat_id))
        original_pause(chat_id)
        pause_seen.set()

    async def fast_keep_typing(
        chat_id: str,
        interval: float = 2.0,
        metadata=None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        # Reuse the real base loop, but shorten the cadence so the test proves
        # the lifecycle shape quickly instead of waiting multiple seconds.
        await BasePlatformAdapter._keep_typing(
            adapter,
            chat_id,
            interval=0.05,
            metadata=metadata,
            stop_event=stop_event,
        )

    async def handler(_event: MessageEvent) -> str:
        return "Background process finished successfully."

    adapter._message_handler = handler
    adapter._bot = SimpleNamespace(
        send_message=mock_send_message,
        send_chat_action=mock_send_typing,
    )
    adapter.stop_typing = mock_stop_typing
    adapter.pause_typing_for_chat = recording_pause
    adapter._keep_typing = fast_keep_typing

    processing_task = None
    try:
        await adapter.handle_message(event)
        processing_task = adapter._session_tasks[session_key]

        await asyncio.wait_for(final_attempt_started.wait(), timeout=1.0)
        await asyncio.wait_for(pause_seen.wait(), timeout=1.0)

        assert attempts["count"] == 3, (
            "the background turn should still be parked in TelegramAdapter.send()'s "
            "final attempt after the retry_after backoffs have already happened"
        )
        assert processing_task is not None and not processing_task.done(), (
            "the message-processing task should still be alive while the final "
            "Telegram send attempt has not returned"
        )
        assert paused_chat_ids == [event.source.chat_id], (
            "Telegram flood control should pause typing for the chat exactly once"
        )
        count_before = len(typing_calls)
        await asyncio.sleep(0.20)
        count_after = len(typing_calls)
        assert count_after == count_before, (
            "typing should stop refreshing once Telegram send enters retry_after "
            "handling; otherwise the bubble can persist indefinitely"
        )
        assert event.source.chat_id in adapter._typing_paused
        assert not stop_typing_called.is_set(), (
            "the outer finally cleanup still should not run before the blocked send returns"
        )
        assert session_key in adapter._active_sessions

        release_final_attempt.set()
        await asyncio.wait_for(processing_task, timeout=1.0)

        assert stop_typing_called.is_set(), (
            "once the blocked send returns, the processing finally-block should "
            "run and stop typing"
        )
        assert event.source.chat_id not in adapter._typing_paused
        assert session_key not in adapter._active_sessions
    finally:
        release_final_attempt.set()
        if processing_task is not None and not processing_task.done():
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass
        await adapter.cancel_background_tasks()
