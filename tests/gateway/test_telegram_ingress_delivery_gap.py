"""Telegram ingress dispatch accounting (#102260).

The transport probes prove getUpdates round-trips complete; these pin the one signal they cannot
give — whether PTB's dispatcher hands the fetched updates to a handler — and the once-per-adapter
report for an adapter with no gateway message handler at all.
"""
import logging
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, Platform, SessionSource
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


@pytest.mark.asyncio
async def test_stall_reported_once_on_backlog_regardless_of_update_age(caplog):
    """Healthy dispatch never reports; a wedged dispatcher is reported after two heartbeats even
    while new updates keep arriving, and only once per stall."""
    adapter = _polling_adapter()
    caplog.set_level(logging.WARNING)
    _receive(adapter, 2)
    await _dispatch(adapter, 2)
    _heartbeats(adapter, 3)
    assert _deaf_reports(caplog) == []

    for _ in range(3):  # a fresh update lands before every heartbeat, none dispatched
        _receive(adapter, 1)
        adapter._check_ingress_dispatch_stall()
    (report,) = _deaf_reports(caplog)
    assert "2 update(s) fetched" in report and "4 received, 2 dispatched" in report


@pytest.mark.asyncio
async def test_dispatch_progress_rearms_the_report(caplog):
    adapter = _polling_adapter()
    caplog.set_level(logging.WARNING)
    _receive(adapter, 3)
    _heartbeats(adapter, 3)
    assert len(_deaf_reports(caplog)) == 1

    await _dispatch(adapter, 1)  # partial drain: progress, backlog remains
    adapter._check_ingress_dispatch_stall()
    assert len(_deaf_reports(caplog)) == 1
    _heartbeats(adapter, 2)
    assert len(_deaf_reports(caplog)) == 2


def test_new_generation_restarts_backlog_and_ignores_fenced_polls(caplog):
    adapter = _polling_adapter()
    caplog.set_level(logging.WARNING)
    _receive(adapter, 3)
    stale_generation = adapter._polling_generation
    adapter._begin_polling_generation()
    assert adapter._record_polling_progress(stale_generation) is False
    _receive(adapter, 5, generation=stale_generation)
    assert adapter._updates_received_total == 0
    _heartbeats(adapter, 3)
    assert _deaf_reports(caplog) == []


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
