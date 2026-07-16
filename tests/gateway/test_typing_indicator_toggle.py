"""Per-platform typing-indicator toggle (PlatformConfig.typing_indicator).

The "typing…" / "is thinking…" status bubble is driven by the generic
``_keep_typing`` refresh loop that ``_process_message_background`` spawns for
every inbound message on every platform.  ``typing_indicator`` (default True)
gates that spawn: when False, the loop is never started, so ``send_typing``
is never called and no status indicator is shown — while message delivery is
otherwise unchanged.

These are behavioral tests against the real dispatch path, not snapshots.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter(typing_indicator: bool) -> _StubAdapter:
    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="t", typing_indicator=typing_indicator),
        Platform.SLACK,
    )
    # Record send_typing calls without performing any platform I/O.
    adapter.send_typing = AsyncMock(return_value=None)
    adapter._send_with_retry = AsyncMock(return_value=None)
    # Handler returns immediately; the typing loop only fires if it was spawned.
    adapter._message_handler = AsyncMock(return_value="ok")
    return adapter


def _make_event(chat_id="C123"):
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id=chat_id, chat_type="dm"),
    )


def _sk(chat_id="C123"):
    return build_session_key(
        SessionSource(platform=Platform.SLACK, chat_id=chat_id, chat_type="dm")
    )


@pytest.mark.asyncio
async def test_typing_indicator_enabled_spawns_refresh_loop():
    """Default (typing_indicator=True): the refresh loop calls send_typing."""
    adapter = _make_adapter(typing_indicator=True)

    # Real handlers take time (tool calls); yield long enough for the spawned
    # refresh loop to fire at least one send_typing before delivery completes.
    async def _slow_handler(_event):
        await asyncio.sleep(0.05)
        return "ok"

    adapter._message_handler = _slow_handler
    event = _make_event()
    adapter._active_sessions[_sk()] = asyncio.Event()

    await adapter._process_message_background(event, _sk())

    assert adapter.send_typing.await_count >= 1


@pytest.mark.asyncio
async def test_typing_indicator_disabled_never_calls_send_typing():
    """typing_indicator=False: the loop is never spawned, send_typing unused."""
    adapter = _make_adapter(typing_indicator=False)
    event = _make_event()
    adapter._active_sessions[_sk()] = asyncio.Event()

    await adapter._process_message_background(event, _sk())

    adapter.send_typing.assert_not_awaited()
    # Delivery still happened — disabling typing must not suppress the reply.
    adapter._send_with_retry.assert_awaited()


@pytest.mark.asyncio
async def test_typing_refresh_stops_before_final_delivery():
    """The refresh loop cannot re-arm typing after final delivery starts."""
    adapter = _make_adapter(typing_indicator=True)
    typing_started = asyncio.Event()
    final_delivery_started = asyncio.Event()
    typing_calls = {"before_delivery": 0, "after_delivery": 0}

    async def _record_typing(*_args, **_kwargs):
        phase = (
            "after_delivery" if final_delivery_started.is_set() else "before_delivery"
        )
        typing_calls[phase] += 1
        typing_started.set()

    async def _fast_keep_typing(chat_id, metadata=None, stop_event=None):
        while stop_event is None or not stop_event.is_set():
            await adapter.send_typing(chat_id, metadata=metadata)
            await asyncio.sleep(0)

    async def _handler(_event):
        await asyncio.wait_for(typing_started.wait(), timeout=1.0)
        return "All done."

    async def _slow_final_delivery(**_kwargs):
        final_delivery_started.set()
        # Give a still-live refresh task ample opportunity to race the final
        # send and re-arm Telegram's typing indicator afterward.
        await asyncio.sleep(0.02)
        return SendResult(success=True, message_id="final")

    adapter.send_typing.side_effect = _record_typing
    adapter._keep_typing = _fast_keep_typing
    adapter._message_handler = _handler
    adapter._send_with_retry.side_effect = _slow_final_delivery
    event = _make_event()
    adapter._active_sessions[_sk()] = asyncio.Event()

    await adapter._process_message_background(event, _sk())

    assert typing_calls["before_delivery"] >= 1
    assert typing_calls["after_delivery"] == 0


@pytest.mark.asyncio
async def test_cancelled_processor_does_not_deliver_after_typing_stop(tmp_path):
    adapter = _make_adapter(typing_indicator=True)
    typing_cleanup_started = asyncio.Event()
    release_typing_cleanup = asyncio.Event()

    attachment = tmp_path / "result.pdf"
    attachment.write_bytes(b"test attachment")

    async def _handler(_event):
        await asyncio.sleep(0)
        return f"Stale final response\nMEDIA:{attachment}"

    async def _slow_typing_refresh(chat_id, metadata=None, stop_event=None):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            typing_cleanup_started.set()
            await release_typing_cleanup.wait()
            raise

    adapter._message_handler = _handler
    adapter._keep_typing = _slow_typing_refresh
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="text")
    )
    adapter.send_document = AsyncMock(
        return_value=SendResult(success=True, message_id="document")
    )
    adapter.filter_media_delivery_paths = lambda paths: paths
    adapter._get_human_delay = lambda: 0

    event = _make_event()
    session_key = _sk()
    adapter._active_sessions[session_key] = asyncio.Event()

    task = asyncio.create_task(
        adapter._process_message_background(event, session_key)
    )
    adapter._session_tasks[session_key] = task

    await asyncio.wait_for(typing_cleanup_started.wait(), timeout=1.0)

    task.cancel()
    release_typing_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    adapter._send_with_retry.assert_not_awaited()
    adapter.send_document.assert_not_awaited()
