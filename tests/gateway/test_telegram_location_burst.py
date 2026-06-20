# FILE: tests/gateway/test_telegram_location_burst.py | PURPOSE: Regression tests for Telegram location bursts so live location updates do not interrupt agents repeatedly.

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.platforms.telegram import TelegramAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _adapter(*, ignore_locations: bool = False):
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    setattr(adapter, "config", SimpleNamespace(extra={"ignore_locations": ignore_locations}))
    adapter._location_live_update_suppression_seconds = 300.0
    adapter._recent_live_locations = {}
    adapter._should_process_message = cast(Any, lambda message, *args, **kwargs: True)
    adapter._should_observe_unmentioned_group_message = cast(Any, lambda message: False)
    adapter._observe_unmentioned_group_message = lambda *args, **kwargs: None
    adapter._build_message_event = cast(
        Any,
        lambda message, msg_type, update_id=None: SimpleNamespace(text="", source=SimpleNamespace()),
    )
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    return adapter


def _location_update(
    *,
    message_id: int,
    lat: float = 52.0,
    lon: float = 13.0,
    live_period=None,
    edited=False,
):
    location = SimpleNamespace(latitude=lat, longitude=lon)
    if live_period is not None:
        location.live_period = live_period
    message = SimpleNamespace(
        location=location,
        venue=None,
        chat_id=2141906339,
        chat=SimpleNamespace(id=2141906339),
        from_user=SimpleNamespace(id=2141906339),
        message_id=message_id,
    )
    return SimpleNamespace(
        message=None if edited else message,
        edited_message=message if edited else None,
        effective_message=message,
        update_id=message_id,
    )


def test_static_location_pin_is_processed_once():
    adapter = _adapter()
    calls = []

    async def handle(event):
        calls.append(event.text)

    adapter.handle_message = handle
    asyncio.run(adapter._handle_location_message(_location_update(message_id=1), None))

    assert len(calls) == 1


def test_location_burst_from_same_chat_user_is_suppressed():
    adapter = _adapter()
    calls = []

    async def handle(event):
        calls.append(event.text)

    adapter.handle_message = handle
    asyncio.run(adapter._handle_location_message(_location_update(message_id=1, live_period=600), None))
    asyncio.run(adapter._handle_location_message(_location_update(message_id=2, lat=52.1, lon=13.1), None))
    asyncio.run(adapter._handle_location_message(_location_update(message_id=1, live_period=600, edited=True), None))

    assert len(calls) == 1


def test_location_messages_can_be_globally_ignored():
    adapter = _adapter(ignore_locations=True)
    calls = []

    async def handle(event):
        calls.append(event.text)

    adapter.handle_message = handle
    asyncio.run(adapter._handle_location_message(_location_update(message_id=1, live_period=600), None))
    asyncio.run(adapter._handle_location_message(_location_update(message_id=2, lat=52.1, lon=13.1), None))

    assert calls == []


def test_gateway_ingress_drops_generated_location_text_before_session_lookup():
    runner = GatewayRunner.__new__(GatewayRunner)
    event = MessageEvent(
        text="[The user shared a location pin.]\nlatitude: 52.478236\nlongitude: 13.36116",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="2141906339",
            chat_type="dm",
            user_id="2141906339",
            user_name="Justin Frederick Schafer",
        ),
    )

    assert asyncio.run(GatewayRunner._handle_message(runner, event)) is None
    assert not hasattr(runner, "session_store")
