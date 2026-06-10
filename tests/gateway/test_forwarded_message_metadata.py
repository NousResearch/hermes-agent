"""Tests for forwarded-message metadata preservation.

Forwarded messages carry platform-native metadata (origin, sender, date)
that adapters extract and pass through as ``MessageEvent.forward_origin``.
The gateway renders this into agent-visible text so the agent can
distinguish forwarded content from user-typed text.
"""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_name="DM",
        chat_type="private",
        user_name="Alice",
    )


@pytest.mark.asyncio
async def test_forward_user_injected():
    """Forwarded from a known user: shows sender name."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="Check this out",
        source=source,
        forward_origin={
            "type": "user",
            "sender_name": "Bob Smith",
            "sender_id": "98765",
            "date": "2026-06-10T09:00:00+00:00",
        },
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "[Forwarded message]" in result
    assert "From: Bob Smith" in result
    assert "Date: 2026-06-10T09:00:00+00:00" in result
    assert result.endswith("Check this out")


@pytest.mark.asyncio
async def test_forward_hidden_user():
    """Forwarded from a hidden user: shows 'hidden sender'."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="Anonymous tip",
        source=source,
        forward_origin={
            "type": "hidden_user",
            "sender_name": "Anonymous",
        },
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "[Forwarded message]" in result
    assert "From: Anonymous" in result
    assert result.endswith("Anonymous tip")


@pytest.mark.asyncio
async def test_forward_hidden_user_no_name():
    """Hidden user without sender_name: shows generic 'hidden sender'."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="Secret message",
        source=source,
        forward_origin={"type": "hidden_user"},
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "From: hidden sender" in result


@pytest.mark.asyncio
async def test_forward_channel():
    """Forwarded from a channel: shows channel name."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="Breaking news",
        source=source,
        forward_origin={
            "type": "channel",
            "chat_name": "Tech News",
            "sender_id": "-100123456",
            "date": "2026-06-10T08:30:00+00:00",
        },
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "[Forwarded message]" in result
    assert "Chat: Tech News" in result
    assert result.endswith("Breaking news")


@pytest.mark.asyncio
async def test_forward_chat():
    """Forwarded from a group chat: shows chat name."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="Team update",
        source=source,
        forward_origin={
            "type": "chat",
            "chat_name": "Engineering Team",
            "sender_id": "-100999",
        },
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "[Forwarded message]" in result
    assert "Chat: Engineering Team" in result


@pytest.mark.asyncio
async def test_no_forward_no_prefix():
    """Ordinary (non-forwarded) messages get no forward prefix."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="Hello world",
        source=source,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "[Forwarded message]" not in result
    assert result == "Hello world"


@pytest.mark.asyncio
async def test_forward_none_no_prefix():
    """Explicit forward_origin=None should not inject prefix."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="Direct message",
        source=source,
        forward_origin=None,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "[Forwarded message]" not in result


@pytest.mark.asyncio
async def test_forward_with_reply_both_prefixes():
    """Forwarded + reply: both prefixes should appear."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="See this",
        source=source,
        forward_origin={
            "type": "user",
            "sender_name": "Charlie",
        },
        reply_to_message_id="10",
        reply_to_text="Previous context",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "[Forwarded message]" in result
    assert "From: Charlie" in result
    assert '[Replying to: "Previous context"]' in result
    assert result.endswith("See this")


@pytest.mark.asyncio
async def test_forward_in_shared_group():
    """Forwarded message in a group: forward prefix appears regardless of group config."""
    runner = _make_runner()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100111",
        chat_name="Group",
        chat_type="group",
        user_name="Dave",
    )
    event = MessageEvent(
        text="Look at this",
        source=source,
        forward_origin={
            "type": "user",
            "sender_name": "Eve",
        },
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "[Forwarded message]" in result
    assert "From: Eve" in result
    assert result.endswith("Look at this")


@pytest.mark.asyncio
async def test_forward_origin_with_no_type():
    """forward_origin dict without 'type' key should be ignored."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="Weird message",
        source=source,
        forward_origin={"some_key": "some_value"},
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "[Forwarded message]" not in result


def test_message_event_forward_origin_field():
    """MessageEvent.forward_origin defaults to None."""
    event = MessageEvent(text="test")
    assert event.forward_origin is None


def test_message_event_forward_origin_settable():
    """MessageEvent.forward_origin can be set to a dict."""
    fwd = {"type": "user", "sender_name": "Test"}
    event = MessageEvent(text="test", forward_origin=fwd)
    assert event.forward_origin == fwd
    assert event.forward_origin["type"] == "user"
