"""Telegram ingress delivery accounting (#102260).

Every existing Telegram health probe measures the *transport*: a getUpdates
round-trip that returns 200 proves bytes are moving and nothing else. When
updates arrive and then die downstream — a dispatcher that never drains, a
filter that never matches, an adapter with no gateway handler installed — the
gateway publishes ``connected``, logs nothing at all, and is indistinguishable
from an idle bot. That is #102260: three weeks of a bot reporting
``telegram.state: "connected"`` and ``polling confirmed healthy`` while not one
inbound message reached a handler.

These tests pin the three counters that split the failure into causes:

* ``_updates_received_total``   — updates Telegram handed the process.
* ``_updates_dispatched_total`` — updates PTB carried through the handler chain.
* ``_inbound_delivered_total``  — inbound events that reached the gateway.
"""
import asyncio
import logging
import time as _time
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, Platform, SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._webhook_mode = False
    adapter._app = MagicMock()
    adapter._app.updater.running = True
    return adapter


def _envelope_request(result):
    """A PTB request double whose parser returns a getUpdates envelope."""
    request = MagicMock()
    request.parse_json_payload = MagicMock(
        return_value={"ok": True, "result": result}
    )
    return request


# ---------------------------------------------------------------------------
# received counter
# ---------------------------------------------------------------------------


def test_empty_getupdates_result_counts_no_updates():
    """An idle long-poll proves the transport, not that anything arrived."""
    adapter = _make_adapter()
    adapter._polling_generation = 1
    adapter._polling_progress_accepting = True
    adapter._polling_progress_event = asyncio.Event()

    adapter._observe_polling_request_result(_envelope_request([]), 1, (200, b"{}"))

    assert adapter._updates_received_total == 0
    assert adapter._last_update_received_monotonic is None
    # ...while transport progress is still recorded.
    assert adapter._polling_last_progress_monotonic is not None


def test_non_empty_getupdates_result_counts_every_update():
    adapter = _make_adapter()
    adapter._polling_generation = 1
    adapter._polling_progress_accepting = True
    adapter._polling_progress_event = asyncio.Event()

    adapter._observe_polling_request_result(
        _envelope_request([{"update_id": 1}, {"update_id": 2}]), 1, (200, b"{}")
    )

    assert adapter._updates_received_total == 2
    assert adapter._last_update_received_monotonic is not None


# ---------------------------------------------------------------------------
# delivered counter
# ---------------------------------------------------------------------------


def _text_event() -> MessageEvent:
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="6053106869",
            user_id="6053106869",
            chat_type="dm",
        ),
    )


@pytest.mark.asyncio
async def test_missing_message_handler_is_logged_once_not_silent(caplog):
    """A connected adapter with no handler discards 100% of inbound.

    Before #102260 this returned with no log line at all, so a mis-wired
    adapter looked exactly like a quiet chat.
    """
    adapter = _make_adapter()
    adapter._message_handler = None

    with caplog.at_level(logging.ERROR):
        await adapter.handle_message(_text_event())
        await adapter.handle_message(_text_event())

    matches = [r for r in caplog.records if "no gateway message handler" in r.message]
    assert len(matches) == 1, "must report the deaf adapter exactly once"
    assert adapter._inbound_delivered_total == 0


@pytest.mark.asyncio
async def test_delivery_to_gateway_handler_increments_counter():
    adapter = _make_adapter()
    adapter.note_inbound_delivered()

    assert adapter._inbound_delivered_total == 1
    assert adapter._last_inbound_delivered_monotonic is not None


# ---------------------------------------------------------------------------
# gap watchdog
# ---------------------------------------------------------------------------


def _armed_gap_adapter(*, received=3, dispatched=3, delivered=0, age=600.0):
    adapter = _make_adapter()
    now = _time.monotonic()
    adapter._updates_received_total = received
    adapter._updates_dispatched_total = dispatched
    adapter._inbound_delivered_total = delivered
    adapter._last_update_received_monotonic = now - age
    adapter._last_inbound_delivered_monotonic = None
    return adapter


def test_no_report_while_within_the_gap_window(caplog):
    adapter = _armed_gap_adapter(age=10.0)
    with caplog.at_level(logging.ERROR):
        adapter._check_ingress_delivery_gap()
    assert not [r for r in caplog.records if "healthy but deaf" in r.message]


def test_no_report_when_nothing_was_ever_received(caplog):
    """An idle bot must never be accused of being deaf."""
    adapter = _make_adapter()
    with caplog.at_level(logging.ERROR):
        adapter._check_ingress_delivery_gap()
    assert not [r for r in caplog.records if "healthy but deaf" in r.message]


def test_no_report_when_delivery_followed_the_last_update(caplog):
    adapter = _armed_gap_adapter()
    adapter._inbound_delivered_total = 3
    adapter._last_inbound_delivered_monotonic = (
        adapter._last_update_received_monotonic + 1
    )
    with caplog.at_level(logging.ERROR):
        adapter._check_ingress_delivery_gap()
    assert not [r for r in caplog.records if "healthy but deaf" in r.message]


def test_dispatched_shortfall_names_the_dispatcher(caplog):
    """received > dispatched: PTB never carried the updates to the handlers."""
    adapter = _armed_gap_adapter(received=5, dispatched=1)
    with caplog.at_level(logging.ERROR):
        adapter._check_ingress_delivery_gap()
    (record,) = [r for r in caplog.records if "healthy but deaf" in r.message]
    assert "dispatcher is not draining" in record.getMessage()


def test_delivery_shortfall_names_hermes(caplog):
    """dispatched == received but nothing delivered: Hermes drops them."""
    adapter = _armed_gap_adapter(received=4, dispatched=4)
    with caplog.at_level(logging.ERROR):
        adapter._check_ingress_delivery_gap()
    (record,) = [r for r in caplog.records if "healthy but deaf" in r.message]
    message = record.getMessage()
    assert "dropped inside Hermes" in message
    assert "4 update(s) received" in message


def test_report_is_not_repeated_until_more_updates_arrive(caplog):
    adapter = _armed_gap_adapter()
    with caplog.at_level(logging.ERROR):
        adapter._check_ingress_delivery_gap()
        adapter._check_ingress_delivery_gap()
    assert len([r for r in caplog.records if "healthy but deaf" in r.message]) == 1

    # A further update with still no delivery is a new, reportable data point.
    adapter._updates_received_total += 1
    adapter._updates_dispatched_total += 1
    with caplog.at_level(logging.ERROR):
        adapter._check_ingress_delivery_gap()
    assert len([r for r in caplog.records if "healthy but deaf" in r.message]) == 2


def test_gap_check_never_triggers_recovery():
    """Reconnecting a healthy transport cannot fix a dropped update.

    The reconnect ladder belongs to the transport probes; this check is
    diagnosis only, so it must leave the ladder untouched.
    """
    adapter = _armed_gap_adapter()
    adapter._check_ingress_delivery_gap()
    assert adapter._polling_error_task is None


def test_gap_check_skipped_in_webhook_mode(caplog):
    adapter = _armed_gap_adapter()
    adapter._webhook_mode = True
    with caplog.at_level(logging.ERROR):
        adapter._check_ingress_delivery_gap()
    assert not [r for r in caplog.records if "healthy but deaf" in r.message]


def test_gap_check_skipped_during_teardown(caplog):
    adapter = _armed_gap_adapter()
    adapter._polling_teardown_started = True
    with caplog.at_level(logging.ERROR):
        adapter._check_ingress_delivery_gap()
    assert not [r for r in caplog.records if "healthy but deaf" in r.message]


# ---------------------------------------------------------------------------
# dispatched counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_catch_all_observer_counts_dispatch_before_early_return():
    """The counter must be stamped even when no plugin hook is installed.

    ``_on_platform_update`` returns immediately without a platform-event
    handler; if the counter sat after that return, every gateway without the
    hook would report a permanent dispatcher stall.
    """
    adapter = _make_adapter()
    adapter._platform_event_handler = None

    await adapter._on_platform_update(MagicMock(), MagicMock())

    assert adapter._updates_dispatched_total == 1


@pytest.mark.asyncio
async def test_heartbeat_runs_the_gap_check():
    """The check has to be wired into the loop that actually runs."""
    adapter = _make_adapter()
    bot = MagicMock()
    bot.get_me = AsyncMock()
    adapter._app.bot = bot
    adapter._bot = bot
    called = []
    adapter._check_ingress_delivery_gap = lambda: called.append(True)
    adapter._probe_pending_updates = AsyncMock()
    adapter._check_polling_stall = AsyncMock()

    real_sleep = asyncio.sleep

    async def fast_sleep(_delay, *args, **kwargs):
        await real_sleep(0)

    task = asyncio.get_running_loop().create_task(adapter._polling_heartbeat_loop())
    import plugins.platforms.telegram.adapter as tg_adapter

    original = tg_adapter.asyncio.sleep
    tg_adapter.asyncio.sleep = fast_sleep
    try:
        for _ in range(50):
            await real_sleep(0)
            if called:
                break
    finally:
        tg_adapter.asyncio.sleep = original
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert called, "_polling_heartbeat_loop must run the ingress gap check"
