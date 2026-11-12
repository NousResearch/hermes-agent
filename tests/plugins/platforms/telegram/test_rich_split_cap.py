"""Rich (``sendRichMessage``) delivery obeys the same per-reply cap as the MarkdownV2 split (#152012).

A 415k rich-eligible reply used to open one rich message per 32,768-char part (13 of them). The parts
are now capped like the legacy split: at most ``MAX_SPLIT_MESSAGES`` (or the stream consumer's
``_split_budget``) messages, the localized notice last, ``split_capped`` reported, and a MarkdownV2
fallback for a refused part stays inside the slots the landed parts left.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture(autouse=True)
def _parse_mode(monkeypatch):
    monkeypatch.setattr(telegram_adapter, "ParseMode", SimpleNamespace(MARKDOWN_V2="MarkdownV2"))


class _BadRequest(Exception):
    pass


def _content(chars: int) -> str:
    rows = "\n".join(f"| {i} | the build stayed green on main, row {i} |" for i in range(chars // 48))
    return "| n | note |\n|---|---|\n" + rows


def _adapter(*, refuse: int | None = None):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={"rich_messages": True}))
    screen: list[tuple[str, str]] = []
    seen = {"rich": 0}

    async def do_api_request(method, api_kwargs=None):
        if method == "sendRichMessage":
            seen["rich"] += 1
            if seen["rich"] == refuse:
                raise _BadRequest("Bad Request: can't parse rich message")
            screen.append(("rich", api_kwargs["rich_message"]["markdown"]))
            return SimpleNamespace(message_id=500 + len(screen))
        return True

    async def send_message(text="", **_kw):
        screen.append(("legacy", text))
        return SimpleNamespace(message_id=500 + len(screen))

    bot = MagicMock()
    bot.do_api_request = AsyncMock(side_effect=do_api_request)
    bot.send_message = AsyncMock(side_effect=send_message)
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot
    return adapter, screen


@pytest.mark.asyncio
async def test_a_huge_rich_reply_opens_at_most_the_cap_ending_with_the_notice():
    adapter, screen = _adapter()
    result = await adapter.send("4242", _content(415_000))

    assert result.success is True
    assert len(screen) == adapter.MAX_SPLIT_MESSAGES
    assert all(kind == "rich" for kind, _ in screen)
    assert "Response truncated" in screen[-1][1]
    assert result.raw_response["split_capped"] is True and result.raw_response["dropped_chars"] > 100_000


@pytest.mark.asyncio
async def test_the_stream_consumer_budget_lowers_the_rich_cap():
    adapter, screen = _adapter()
    result = await adapter.send("4242", _content(415_000), metadata={"_split_budget": 3})

    assert len(screen) == 3 and "Response truncated" in screen[-1][1]
    assert result.raw_response["split_capped"] is True


@pytest.mark.asyncio
async def test_warnings_off_drops_the_rich_tail_without_a_notice(monkeypatch):
    adapter, screen = _adapter()
    monkeypatch.setattr(adapter, "warning_notifications_enabled", lambda *a, **k: False)
    result = await adapter.send("4242", _content(415_000))

    assert len(screen) == adapter.MAX_SPLIT_MESSAGES - 1
    assert not any("Response truncated" in text for _, text in screen)
    assert result.raw_response["split_capped"] is True


@pytest.mark.asyncio
async def test_a_refused_part_falls_back_inside_the_remaining_budget(monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", AsyncMock())
    adapter, screen = _adapter(refuse=3)
    result = await adapter._send_with_retry("4242", _content(415_000))

    assert len(screen) <= adapter.MAX_SPLIT_MESSAGES
    assert [kind for kind, _ in screen[:2]] == ["rich", "rich"]
    assert "Response truncated" in screen[-1][1].replace("\\", "")
    assert result.success is True and result.raw_response["split_capped"] is True


@pytest.mark.asyncio
async def test_a_rich_final_edit_is_capped():
    adapter, screen = _adapter()
    result = await adapter.edit_message("4242", "100", _content(415_000), finalize=True)

    # The edited preview plus its continuations: never more than the cap.
    assert 1 + len(screen) == adapter.MAX_SPLIT_MESSAGES
    assert "Response truncated" in screen[-1][1]
    assert result.raw_response["split_capped"] is True


@pytest.mark.asyncio
async def test_a_rich_reply_under_the_cap_is_untouched():
    adapter, screen = _adapter()
    result = await adapter.send("4242", _content(40_000))

    assert len(screen) == 2 and not any("Response truncated" in text for _, text in screen)
    assert "split_capped" not in result.raw_response
