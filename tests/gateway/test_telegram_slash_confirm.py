"""Regression guard: send_slash_confirm must use format_message + MARKDOWN_V2."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter
from gateway.config import PlatformConfig


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestSendSlashConfirm:

    @pytest.mark.asyncio
    async def test_uses_markdown_v2_and_escapes_special_chars(self):
        """send_slash_confirm must pass preview through format_message and use
        MARKDOWN_V2 — so commands with underscores, dots, or brackets don't
        raise BadRequest: Can't parse entities."""
        adapter = _make_adapter()
        sent = {}

        async def mock_send(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=7)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send)

        result = await adapter.send_slash_confirm(
            chat_id="100",
            title="Confirm",
            message="/run script_name.sh --flag=value [option]",
            session_key="sk",
            confirm_id="cid1",
        )

        assert result.success is True
        assert "MARKDOWN_V2" in repr(sent["parse_mode"])
        # Underscores and dots must be escaped by format_message
        assert "script\\_name" in sent["text"]
        assert "\\." in sent["text"]

    @pytest.mark.asyncio
    async def test_stores_slash_confirm_state(self):
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(
            return_value=SimpleNamespace(message_id=8)
        )

        await adapter.send_slash_confirm(
            chat_id="100",
            title="Confirm",
            message="reload-mcp",
            session_key="my-session",
            confirm_id="cid2",
        )

        assert adapter._slash_confirm_state["cid2"] == "my-session"

    @pytest.mark.asyncio
    async def test_not_connected_returns_failure(self):
        adapter = _make_adapter()
        adapter._bot = None

        result = await adapter.send_slash_confirm(
            chat_id="100",
            title="Confirm",
            message="reload-mcp",
            session_key="sk",
            confirm_id="cid3",
        )

        assert result.success is False
