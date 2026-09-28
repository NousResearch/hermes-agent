"""TelegramAdapter.set_message_buttons — backs ctx.platform_actions.set_message_buttons (#64176)."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_sets_one_button_per_row_on_the_message():
    adapter = _make_adapter()
    ok = await adapter.set_message_buttons("-1001", "42", [{"label": "Apply", "data": "p:apply"},
                                                          {"label": "Skip", "data": "p:skip"}])
    assert ok is True
    kwargs = adapter._bot.edit_message_reply_markup.await_args.kwargs
    assert kwargs["message_id"] == 42
    assert kwargs["chat_id"] == tg.normalize_telegram_chat_id("-1001")
    assert kwargs["reply_markup"] is not None


@pytest.mark.asyncio
async def test_empty_list_removes_the_keyboard():
    adapter = _make_adapter()
    assert await adapter.set_message_buttons("1", "2", []) is True
    assert adapter._bot.edit_message_reply_markup.await_args.kwargs["reply_markup"] is None


@pytest.mark.asyncio
async def test_not_modified_counts_as_success_other_errors_do_not():
    adapter = _make_adapter()
    adapter._bot.edit_message_reply_markup.side_effect = Exception("Bad Request: message is not modified")
    assert await adapter.set_message_buttons("1", "2", [{"label": "a", "data": "p:a"}]) is True
    adapter._bot.edit_message_reply_markup.side_effect = Exception("Bad Request: message to edit not found")
    assert await adapter.set_message_buttons("1", "2", [{"label": "a", "data": "p:a"}]) is False


@pytest.mark.asyncio
async def test_disconnected_adapter_returns_false():
    adapter = _make_adapter()
    adapter._bot = None
    assert await adapter.set_message_buttons("1", "2", []) is False
