"""Telegram ingress dispatch accounting (#102260, #130407).

The transport probes prove getUpdates round-trips complete; these pin the one signal they cannot
give — whether PTB's dispatcher hands the fetched updates to a handler — and the once-per-adapter
report for an adapter with no gateway message handler at all.
"""
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, Platform, SessionSource
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter

_DEAF = "healthy but deaf"


def _polling_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._webhook_mode = False
    adapter._begin_polling_generation()
    return adapter


def _receive(adapter: TelegramAdapter, n: int, generation: int | None = None) -> None:
    request = MagicMock()
    request.parse_json_payload = MagicMock(
        return_value={"ok": True, "result": [{"update_id": i} for i in range(n)]}
    )
    adapter._observe_polling_request_result(
        request, adapter._polling_generation if generation is None else generation, (200, b"{}")
    )


async def _dispatch(adapter: TelegramAdapter, n: int) -> None:
    for _ in range(n):
        await adapter._on_platform_update(MagicMock(), MagicMock())


def _heartbeats(adapter: TelegramAdapter, n: int) -> None:
    for _ in range(n):
        adapter._check_ingress_dispatch_stall()


def _deaf_reports(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if _DEAF in r.message]


def _stall_reports(sched) -> list[str]:
    """Errors handed to the (stubbed) recovery handoff: the stall's only announcement."""
    errors = [call.args[0] for call in sched.call_args_list]
    assert all(type(e) is tg_adapter._PollingStallError and _DEAF in str(e) for e in errors)
    return [str(e) for e in errors]


@pytest.mark.asyncio
async def test_stall_reported_once_on_backlog_regardless_of_update_age():
    """Healthy dispatch never reports; a wedged dispatcher is reported after four heartbeats even
    while new updates keep arriving, and only once per stall."""
    adapter = _polling_adapter()
    _receive(adapter, 2)
    await _dispatch(adapter, 2)
    with patch.object(adapter, "_schedule_polling_recovery") as sched:
        _heartbeats(adapter, 2 * tg_adapter._INGRESS_DISPATCH_STALL_HEARTBEATS)  # outlast the first-heartbeat re-arm
        assert sched.call_count == 0

        for _ in range(5):  # a fresh update lands before every heartbeat, none dispatched
            _receive(adapter, 1)
            adapter._check_ingress_dispatch_stall()
    (report,) = _stall_reports(sched)
    assert "4 update(s) fetched" in report and "6 received, 2 dispatched" in report


@pytest.mark.asyncio
async def test_dispatch_progress_rearms_the_report(caplog):
    adapter = _polling_adapter()
    caplog.set_level(logging.WARNING)
    with patch.object(adapter, "_schedule_polling_recovery") as sched:
        _receive(adapter, 3)
        _heartbeats(adapter, 3)  # 270s: a slow-but-bounded (<=300s) sequential handler is not a wedge
        assert sched.call_count == 0
        _heartbeats(adapter, 2)
        assert len(_stall_reports(sched)) == 1

        await _dispatch(adapter, 1)  # partial drain: progress, backlog remains
        adapter._check_ingress_dispatch_stall()
        assert len(_stall_reports(sched)) == 1
        _heartbeats(adapter, 4)
        assert len(_stall_reports(sched)) == 2
    assert _deaf_reports(caplog) == []  # stubbed handoff: no separate pre-log announces the stall


@pytest.mark.asyncio
async def test_dispatch_stall_marks_degraded_and_goes_fatal():
    """A confirmed dispatch stall (#130407) is a handoff, not a retry: degraded status,
    retryable fatal, no in-place updater restart, no backoff sleep."""
    adapter = _polling_adapter()
    adapter._running = True
    adapter._mark_degraded = MagicMock()
    updater = MagicMock()
    updater.stop = AsyncMock(return_value=None)
    updater.start_polling = AsyncMock()
    adapter._app = MagicMock()
    adapter._app.updater = updater
    adapter._drain_polling_connections = AsyncMock()
    adapter._notify_fatal_error = AsyncMock()
    _receive(adapter, 2)

    try:
        with patch("asyncio.sleep", new=AsyncMock()) as sleep:
            _heartbeats(adapter, 3)
            assert adapter._polling_error_task is None
            adapter._check_ingress_dispatch_stall()
            task = adapter._polling_error_task
            assert task is not None
            await task
        assert adapter.has_fatal_error
        assert adapter.fatal_error_retryable is True
        assert "PTB dispatcher made no progress" in adapter.fatal_error_message
        adapter._notify_fatal_error.assert_awaited_once()
        adapter._mark_degraded.assert_called_once()
        updater.start_polling.assert_not_awaited()
        updater.stop.assert_not_awaited()
        sleep.assert_not_awaited()
        adapter._drain_polling_connections.assert_not_awaited()
    finally:
        for pending in tuple(adapter._background_tasks):
            pending.cancel()
        await asyncio.gather(*tuple(adapter._background_tasks), return_exceptions=True)


def test_new_generation_restarts_backlog_and_ignores_fenced_polls():
    adapter = _polling_adapter()
    _receive(adapter, 3)
    stale_generation = adapter._polling_generation
    adapter._begin_polling_generation()
    assert adapter._record_polling_progress(stale_generation) is False
    _receive(adapter, 5, generation=stale_generation)
    assert adapter._updates_received_total == 0


@pytest.mark.asyncio
async def test_missing_message_handler_is_logged_once_not_silent(caplog):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._message_handler = None
    event = MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", user_id="1", chat_type="dm"),
    )
    with caplog.at_level(logging.ERROR):
        await adapter.handle_message(event)
        await adapter.handle_message(event)
    assert len([r for r in caplog.records if "no gateway message handler" in r.message]) == 1
