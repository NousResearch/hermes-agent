"""Unit tests for inbound_message_runtime_service (shared inbound prep)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.inbound_message_runtime_service import prepare_inbound_message_text
from gateway.platforms.base import MessageType
from gateway.session import SessionSource


def _source(**kw) -> SessionSource:
    base = dict(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    base.update(kw)
    return SessionSource(**base)


@pytest.mark.asyncio
async def test_prepare_inbound_plain_text():
    runner = SimpleNamespace(
        config=SimpleNamespace(
            group_sessions_per_user=True,
            thread_sessions_per_user=False,
        ),
        _session_key_for_source=MagicMock(return_value="sk"),
        _consume_pending_native_image_paths=MagicMock(),
    )
    event = SimpleNamespace(
        text="hello",
        media_urls=[],
        media_types=[],
        message_type=MessageType.TEXT,
        reply_to_text=None,
        reply_to_message_id=None,
        channel_context=None,
        message_id="1",
    )
    out = await prepare_inbound_message_text(
        runner=runner,
        event=event,
        source=_source(),
        history=[],
        session_key="sk",
        logger=MagicMock(),
    )
    assert out == "hello"
    runner._consume_pending_native_image_paths.assert_called_once_with("sk")


@pytest.mark.asyncio
async def test_prepare_inbound_reply_to_prefix():
    runner = SimpleNamespace(
        config=SimpleNamespace(
            group_sessions_per_user=True,
            thread_sessions_per_user=False,
        ),
        _session_key_for_source=MagicMock(return_value="sk"),
        _consume_pending_native_image_paths=MagicMock(),
    )
    event = SimpleNamespace(
        text="and yours?",
        media_urls=[],
        media_types=[],
        message_type=MessageType.TEXT,
        reply_to_text="I like cats",
        reply_to_message_id="42",
        reply_to_is_own_message=False,
        channel_context=None,
        message_id="2",
    )
    out = await prepare_inbound_message_text(
        runner=runner,
        event=event,
        source=_source(),
        history=[],
        session_key="sk",
        logger=MagicMock(),
    )
    assert out.startswith('[Replying to: "I like cats"]')
    assert "and yours?" in out


@pytest.mark.asyncio
async def test_runner_delegate_still_works(monkeypatch):
    """GatewayRunner method remains the public entry for existing tests."""
    import gateway.run as gateway_run

    called = {}

    async def _fake(**kwargs):
        called.update(kwargs)
        return "delegated"

    monkeypatch.setattr(
        gateway_run,
        "prepare_gateway_inbound_message_text",
        _fake,
    )
    runner = object.__new__(gateway_run.GatewayRunner)
    event = SimpleNamespace(text="x")
    source = _source()
    out = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
        session_key="sk",
    )
    assert out == "delegated"
    assert called["runner"] is runner
    assert called["event"] is event
