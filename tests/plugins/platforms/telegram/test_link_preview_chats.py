"""Per-chat link-preview suppression: ``disable_link_preview_chats`` turns previews off only for the
listed chats (e.g. a news channel), while ``disable_link_previews`` keeps its global meaning."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_ids import link_preview_disabled_chats, link_previews_disabled_for


@pytest.mark.parametrize("raw,expected", [
    (None, frozenset()),
    ("", frozenset()),
    ("-1001, -1002", frozenset({"-1001", "-1002"})),
    ("-1001;@news_channel", frozenset({"-1001", "@news_channel"})),
    (["-1001", -1002, " "], frozenset({"-1001", "-1002"})),
    (-1001, frozenset({"-1001"})),
])
def test_chat_list_parsing(raw, expected):
    extra = {} if raw is None else {"disable_link_preview_chats": raw}
    assert link_preview_disabled_chats(extra) == expected


@pytest.mark.parametrize("extra,chat_id,expected", [
    ({}, "-1001", False),
    (None, "-1001", False),
    ({"disable_link_previews": True}, "-1001", True),
    ({"disable_link_previews": "true"}, 42, True),
    ({"disable_link_previews": False, "disable_link_preview_chats": "-1001"}, "-1001", True),
    ({"disable_link_preview_chats": "-1001"}, -1001, True),
    ({"disable_link_preview_chats": "-1001"}, "-1002", False),
])
def test_disabled_for_chat(extra, chat_id, expected):
    assert link_previews_disabled_for(extra, chat_id) is expected


def _adapter(extra):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra=extra))
    adapter._bot, adapter._app = AsyncMock(), MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    return adapter


def test_adapter_kwargs_only_for_listed_chat():
    adapter = _adapter({"disable_link_preview_chats": "-1001"})
    assert adapter._link_preview_kwargs("-1001")
    assert adapter._link_preview_kwargs(-1001)
    assert adapter._link_preview_kwargs("-1002") == {}
    assert adapter._link_preview_kwargs() == {}


def test_adapter_global_flag_still_covers_every_chat():
    adapter = _adapter({"disable_link_previews": True})
    assert adapter._link_preview_kwargs("-1002")
    assert adapter._link_preview_kwargs()


def test_adapter_send_applies_per_chat_setting():
    adapter = _adapter({"disable_link_preview_chats": "-1001"})
    asyncio.run(adapter.send("-1001", "see https://example.com"))
    listed = adapter._bot.send_message.await_args.kwargs
    asyncio.run(adapter.send("-1002", "see https://example.com"))
    other = adapter._bot.send_message.await_args.kwargs
    assert "link_preview_options" in listed or listed.get("disable_web_page_preview") is True
    assert "link_preview_options" not in other and "disable_web_page_preview" not in other


def test_send_message_tool_passes_per_chat_setting():
    from tools import send_message_tool as smt
    from gateway.config import Platform

    pconfig = SimpleNamespace(token="t", extra={"disable_link_preview_chats": "-1001"})
    seen = []

    async def fake_send(token, chat_id, message, **kwargs):
        seen.append((chat_id, kwargs.get("disable_link_previews")))
        return {"success": True}

    with patch.object(smt, "_send_telegram", fake_send):
        asyncio.run(smt._send_to_platform(Platform.TELEGRAM, pconfig, "-1001", "x"))
        asyncio.run(smt._send_to_platform(Platform.TELEGRAM, pconfig, "-1002", "x"))
    assert seen == [("-1001", True), ("-1002", False)]
