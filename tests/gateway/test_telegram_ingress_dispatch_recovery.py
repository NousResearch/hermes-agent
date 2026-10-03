"""Telegram ingress dispatch stall recovery (#130407).

Section 1 of #130407: getUpdates kept fetching inbound messages while the PTB
Application dispatcher stopped draining its local queue. The adapter logged
"healthy but deaf" yet kept reporting ``connected`` and never initiated
recovery — a manual restart was required.

The fix: :meth:`_check_ingress_dispatch_stall` warns once per stall AND hands
the adapter to the supervisor for a rebuild via ``_schedule_polling_recovery``
with :class:`_IngressDispatchStallError` (a ``_PollingStallError`` subclass, so
the ladder skips the in-place Updater restart — the wedged dispatcher lives on
the same ``Application`` — and goes retryable-fatal). ``_schedule`` also marks
the adapter degraded so ``gateway_state.json`` stops saying ``connected``.

These pin the recovery contract; the pure counting/reporting contract lives in
``test_telegram_ingress_delivery_gap.py`` (which stubs recovery).
"""
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter

_DEAF = "healthy but deaf"


def _polling_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._webhook_mode = False
    adapter._begin_polling_generation()
    return adapter


def _receive(adapter: TelegramAdapter, n: int) -> None:
    request = MagicMock()
    request.parse_json_payload = MagicMock(
        return_value={"ok": True, "result": [{"update_id": i} for i in range(n)]}
    )
    adapter._observe_polling_request_result(
        request, adapter._polling_generation, (200, b"{}")
    )


async def _dispatch(adapter: TelegramAdapter, n: int) -> None:
    for _ in range(n):
        await adapter._on_platform_update(MagicMock(), MagicMock())


def _deaf_reports(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if _DEAF in r.message]


@pytest.mark.asyncio
async def test_dispatch_stall_marks_degraded_and_goes_fatal():
    """A confirmed dispatch stall is a handoff, not a retry: degraded status,
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
            adapter._check_ingress_dispatch_stall()
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


@pytest.mark.asyncio
async def test_dispatch_progress_rearms_recovery_scheduling(caplog):
    """With the handoff stubbed (no fatal), partial dispatch progress re-arms the next escalation."""
    adapter = _polling_adapter()
    caplog.set_level(logging.WARNING)
    _receive(adapter, 3)
    with patch.object(adapter, "_schedule_polling_recovery") as sched:
        adapter._check_ingress_dispatch_stall()
        adapter._check_ingress_dispatch_stall()
        assert sched.call_count == 1
        assert len(_deaf_reports(caplog)) == 1
        await _dispatch(adapter, 1)
        adapter._check_ingress_dispatch_stall()
        assert sched.call_count == 1
        adapter._check_ingress_dispatch_stall()
        adapter._check_ingress_dispatch_stall()
        assert sched.call_count == 2
    assert len(_deaf_reports(caplog)) == 2
    for call in sched.call_args_list:
        (err,) = call.args
        assert isinstance(err, tg_adapter._IngressDispatchStallError)
