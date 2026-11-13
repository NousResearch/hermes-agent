"""The live Telegram gateway must explain a private @username home send (#95121)."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.parametrize(
    ("chat_id", "api_error", "hint_expected"),
    [
        ("@shelmecha", "Chat not found", True),
        ("5843629652", "Chat not found", False),
        ("@shelmecha", "Forbidden: bot was blocked by the user", False),
    ],
)
def test_live_gateway_username_home_error_explains_numeric_id(
    monkeypatch: pytest.MonkeyPatch, chat_id: str, api_error: str, hint_expected: bool,
) -> None:
    """Unlike the CLI, an in-process gateway dispatches through TelegramAdapter.send."""
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._bot = object()

    @asynccontextmanager
    async def no_lock(_self, _chat_id):
        yield

    async def refused(_self, _chat_id, _content, _reply_to, _metadata):
        return SendResult(success=False, error=api_error)

    monkeypatch.setattr(TelegramAdapter, "_chat_send_lock", no_lock)
    monkeypatch.setattr(TelegramAdapter, "_send_text_locked", refused)

    result = asyncio.run(adapter.send(chat_id, "notification"))
    assert result.success is False
    if hint_expected:
        assert "Chat not found" in result.error
        assert "numeric chat id" in result.error
        assert "/sethome" in result.error
    else:
        assert result.error == api_error
