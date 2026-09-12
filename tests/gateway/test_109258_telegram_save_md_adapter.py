"""Regression coverage for Telegram ``/save`` adapter resolution (#109258)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.slash_commands_session import GatewaySessionCommandsMixin


def _event() -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        user_id="user-1",
    )
    return MessageEvent(text="/save md filename.md", source=source)


def _handler(adapter=None) -> GatewaySessionCommandsMixin:
    handler = GatewaySessionCommandsMixin()
    handler.adapters = {} if adapter is None else {Platform.TELEGRAM: adapter}
    handler.async_session_store = SimpleNamespace(
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        )
    )
    handler._session_db = SimpleNamespace(
        export_session=AsyncMock(
            return_value={
                "id": "session-1",
                "messages": [{"role": "user", "content": "Keep this transcript."}],
            }
        )
    )
    return handler


@pytest.mark.asyncio
async def test_telegram_save_md_uses_registered_adapter():
    delivered = {}

    async def send_document(**kwargs):
        delivered.update(kwargs)
        with open(kwargs["file_path"], encoding="utf-8") as exported:
            delivered["content"] = exported.read()

    adapter = SimpleNamespace(send_document=AsyncMock(side_effect=send_document))
    handler = _handler(adapter)

    result = await handler._handle_save_command(_event())

    assert result == "Export complete."
    assert delivered["chat_id"] == "chat-1"
    assert delivered["file_name"] == "filename.md"
    assert delivered["caption"] == "Session export: filename.md"
    assert "Keep this transcript." in delivered["content"]


@pytest.mark.asyncio
async def test_telegram_save_md_reports_absent_adapter():
    handler = _handler()

    result = await handler._handle_save_command(_event())

    assert result == "Platform adapter not found to send the document."
