from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gateway.calls.browser_room import BrowserRoomConfig, BrowserRoomProvider
from gateway.calls.manager import CallManager
from gateway.calls.models import CallState
from gateway.calls.tokens import CallTokenService


def _source(chat_type="dm", platform="telegram"):
    platform = SimpleNamespace(value=platform)
    return SimpleNamespace(platform=platform, chat_id="123", user_id="456", chat_type=chat_type)


@pytest.mark.asyncio
async def test_start_browser_call_rejects_group_chat():
    manager = CallManager(
        browser_provider=BrowserRoomProvider(BrowserRoomConfig(base_url="https://host.ts.net/call")),
        token_service=CallTokenService("secret"),
    )

    result = await manager.start_browser_call(_source(chat_type="group"))

    assert result.ok is False
    assert result.code == "call_private_chat_required"
    assert "private-only" in result.message


@pytest.mark.asyncio
async def test_start_browser_call_creates_tailnet_link():
    manager = CallManager(
        browser_provider=BrowserRoomProvider(BrowserRoomConfig(base_url="https://host.ts.net/call")),
        token_service=CallTokenService("secret"),
        now=lambda: datetime(2026, 5, 26, tzinfo=timezone.utc),
    )

    result = await manager.start_browser_call(_source())

    assert result.ok is True
    assert result.session is not None
    assert result.session.state == CallState.WAITING
    assert result.session.room_url.startswith("https://host.ts.net/call/")
    assert "Private call room ready" in result.message


@pytest.mark.asyncio
async def test_status_reports_idle_without_session():
    manager = CallManager(
        browser_provider=BrowserRoomProvider(BrowserRoomConfig(base_url="https://host.ts.net/call")),
        token_service=CallTokenService("secret"),
    )

    result = await manager.status(_source())

    assert result.ok is True
    assert "No active call" in result.message


@pytest.mark.asyncio
async def test_native_session_status_reports_connecting():
    manager = CallManager(
        browser_provider=BrowserRoomProvider(BrowserRoomConfig(base_url="https://host.ts.net/call")),
        token_service=CallTokenService("secret"),
        now=lambda: datetime(2026, 5, 26, tzinfo=timezone.utc),
    )
    source = _source(platform="simplex")

    session = manager.record_native_call(source, "native-call-1")
    result = await manager.status(source)

    assert session.mode == "simplex_native"
    assert session.state == CallState.CONNECTING
    assert session.room_url is None
    assert result.ok is True
    assert "native-call-1" in result.message
    assert "connecting" in result.message


@pytest.mark.asyncio
async def test_record_native_call_preserves_existing_browser_session():
    manager = CallManager(
        browser_provider=BrowserRoomProvider(BrowserRoomConfig(base_url="https://host.ts.net/call")),
        token_service=CallTokenService("secret"),
    )
    source = _source(platform="simplex")
    browser_result = await manager.start_browser_call(source)

    session = manager.record_native_call(source, "native-call-1")
    status = await manager.status(source)

    assert browser_result.session is not None
    assert session == browser_result.session
    assert session.mode == "browser"
    assert session.call_id != "native-call-1"
    assert "native-call-1" not in status.message


@pytest.mark.asyncio
async def test_end_marks_session_ended():
    manager = CallManager(
        browser_provider=BrowserRoomProvider(BrowserRoomConfig(base_url="https://host.ts.net/call")),
        token_service=CallTokenService("secret"),
    )
    await manager.start_browser_call(_source())

    result = await manager.end(_source())

    assert result.ok is True
    assert "ended" in result.message.lower()
