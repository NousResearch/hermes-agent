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
    adapter._app = MagicMock()
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


def _deaf_reports(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if _DEAF in r.message]


@pytest.mark.asyncio
async def test_dispatch_stall_is_reported_on_backlog_not_update_age(caplog):
    """A wedged dispatcher is reported after two heartbeats even while new updates keep arriving;
    dispatch progress re-arms it; a stale generation cannot inflate the backlog."""
    adapter = _polling_adapter()
    caplog.set_level(logging.WARNING)

    # Healthy: every fetched update reaches the group-99 catch-all.
    _receive(adapter, 2)
    for _ in range(2):
        await adapter._on_platform_update(MagicMock(), MagicMock())
    for _ in range(3):
        adapter._check_ingress_dispatch_stall()
    assert _deaf_reports(caplog) == []

    # Dispatcher wedged: a fresh update lands before every heartbeat, none dispatched.
    _receive(adapter, 1)
    adapter._check_ingress_dispatch_stall()
    assert _deaf_reports(caplog) == [], "one heartbeat is not a stall"
    _receive(adapter, 1)
    adapter._check_ingress_dispatch_stall()
    (report,) = _deaf_reports(caplog)
    assert "2 update(s) fetched" in report and "4 received, 2 dispatched" in report
    _receive(adapter, 1)
    adapter._check_ingress_dispatch_stall()
    assert len(_deaf_reports(caplog)) == 1, "a persistent stall is reported once"

    # Dispatch resumes but only partially drains the backlog: progress re-arms the check, and the
    # next two heartbeats without progress are a new stall.
    await adapter._on_platform_update(MagicMock(), MagicMock())
    adapter._check_ingress_dispatch_stall()
    assert len(_deaf_reports(caplog)) == 1
    for _ in range(2):
        adapter._check_ingress_dispatch_stall()
    assert len(_deaf_reports(caplog)) == 2

    # A rebuilt consumer starts the backlog from zero, and a late response from the fenced
    # generation is ignored.
    stale_generation = adapter._polling_generation
    adapter._begin_polling_generation()
    assert adapter._record_polling_progress(stale_generation) is False
    _receive(adapter, 5, generation=stale_generation)
    assert adapter._updates_received_total == 0
    for _ in range(3):
        adapter._check_ingress_dispatch_stall()
    assert len(_deaf_reports(caplog)) == 2


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
