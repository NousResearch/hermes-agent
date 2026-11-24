"""The live Telegram gateway must explain a private @username home send (#95121).

An in-process gateway dispatches send_message through ``TelegramAdapter.send`` (not
``standalone_send``), so the hint must come from the adapter's own failure boundary. The bot
refuses the way the Bot API does; the real send path (chunking, error classification, the
media fallback notice) runs unmocked.
"""

import asyncio

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _RefusingBot:
    def __init__(self, error: str) -> None:
        self.error = error

    async def send_message(self, **_kw):
        raise RuntimeError(self.error)

    async def send_document(self, **_kw):
        raise RuntimeError(self.error)


def _adapter(error: str) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="1:fake", extra={"rich_messages": False}))
    adapter._bot = _RefusingBot(error)
    return adapter


@pytest.mark.parametrize(
    ("chat_id", "api_error", "hint_expected"),
    [
        ("@shelmecha", "Chat not found", True),
        ("5843629652", "Chat not found", False),
        ("@shelmecha", "Forbidden: bot was blocked by the user", False),
    ],
)
def test_live_gateway_username_home_error_explains_numeric_id(chat_id: str, api_error: str, hint_expected: bool) -> None:
    result = asyncio.run(_adapter(api_error).send(chat_id, "notification"))
    assert result.success is False
    error = result.error or ""
    if hint_expected:
        assert error.startswith("Chat not found.")
        assert "numeric chat id" in error and "/sethome" in error
    else:
        assert error == api_error


def test_live_media_send_to_username_home_carries_the_hint(tmp_path) -> None:
    """send_message with media: a refused document falls back to a text notice through send()."""
    attachment = tmp_path / "report.txt"
    attachment.write_text("x")
    result = asyncio.run(_adapter("Chat not found").send_document("@shelmecha", str(attachment)))
    error = result.error or ""
    assert result.success is False
    assert "numeric chat id" in error and "/sethome" in error
