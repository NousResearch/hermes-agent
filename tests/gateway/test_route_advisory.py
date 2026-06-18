from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(platform=Platform.TELEGRAM):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True)})
    runner._route_advisory_config = {
        "enabled": True,
        "gateway_notice": True,
        "min_confidence": 1.0,
        "cooldown_seconds": 3600.0,
        "platforms": {"telegram"},
    }
    runner._route_advisory_sent = {}
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}  # type: ignore[assignment]
    runner.session_store = None  # type: ignore[assignment]

    def _thread_meta(source, reply_to_message_id=None):
        return {"thread_id": source.thread_id} if source.thread_id else {}

    runner._thread_metadata_for_source = _thread_meta  # type: ignore[method-assign]
    return runner, adapter


def _make_event(text="Prepare BMI ASCAP radio promotion plan", platform=Platform.TELEGRAM):
    source = SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
        thread_id="t1" if platform == Platform.TELEGRAM else None,
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )


@pytest.mark.asyncio
async def test_gateway_route_advisory_sends_telegram_notice(monkeypatch):
    runner, adapter = _make_runner()
    event = _make_event()

    def fake_classify(prompt, *, surface, log):
        assert prompt == event.text
        assert surface == "gateway:telegram"
        assert log is True
        return {
            "route_id": "business-growth",
            "profile": "business-growth",
            "owner": "business-growth",
            "confidence": 4.0,
            "requires_approval": True,
            "blocked_actions": ["unapproved_account_access"],
            "advisory_mode": True,
            "auto_execute": False,
        }

    monkeypatch.setattr("gateway.run.classify_route_advisory", fake_classify)

    advisory = await runner._maybe_send_route_advisory(event, event.source, command=None)

    assert advisory is not None
    assert advisory["route_id"] == "business-growth"
    adapter.send.assert_awaited_once()
    args, kwargs = adapter.send.await_args
    assert args[0] == "c1"
    assert "Route advisory: business-growth -> profile business-growth" in args[1]
    assert "advisory-only" in args[1]
    assert kwargs["metadata"] == {"thread_id": "t1"}


@pytest.mark.asyncio
async def test_gateway_route_advisory_skips_commands(monkeypatch):
    runner, adapter = _make_runner()
    event = _make_event("/status")

    called = {"value": False}

    def fake_classify(*args, **kwargs):
        called["value"] = True
        return {}

    monkeypatch.setattr("gateway.run.classify_route_advisory", fake_classify)

    advisory = await runner._maybe_send_route_advisory(event, event.source, command="status")

    assert advisory is None
    assert called["value"] is False
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_route_advisory_debounces_same_route(monkeypatch):
    runner, adapter = _make_runner()
    event = _make_event()

    monkeypatch.setattr(
        "gateway.run.classify_route_advisory",
        lambda *args, **kwargs: {
            "route_id": "design-media",
            "profile": "design-media",
            "owner": "design-media",
            "confidence": 3.0,
            "requires_approval": True,
            "blocked_actions": ["raw_client_data_upload"],
        },
    )

    await runner._maybe_send_route_advisory(event, event.source, command=None)
    await runner._maybe_send_route_advisory(event, event.source, command=None)

    adapter.send.assert_awaited_once()
