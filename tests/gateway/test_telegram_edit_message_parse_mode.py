"""``TelegramAdapter.edit_message(parse_mode=...)``: caller-formatted edits.

A plugin that draws its own screen (an HTML status board, a monospace table) needs the public edit verb
to send that markup as is. Without the parameter it either loses the formatting (a final edit converts
the text to MarkdownV2 and escapes the tags, an interim edit sends plain text) or reaches for the
private ``_edit_text``. With ``parse_mode`` set the content goes verbatim, is never converted, upgraded
to rich, downgraded to plain, split or truncated, and rejected markup comes back as a failed SendResult.
"""
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.constants import ParseMode

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter, _strip_mdv2

HTML = "<b>CPU</b> 42% · <i>load</i> 0.8"
MARKDOWN = "**done** with `code`"


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._rich_send_disabled = True
    adapter._telegram_chat_outbound_slot_secs = 0.0
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock()
    adapter._bot.edit_message_text = AsyncMock(return_value=MagicMock())
    return adapter


def _last_edit_kwargs(adapter: TelegramAdapter) -> dict:
    return adapter._bot.edit_message_text.await_args.kwargs


def test_edit_message_takes_parse_mode_as_a_keyword_parameter():
    """Plugins detect the capability from the signature, so the name and kind are the contract."""
    parameter = inspect.signature(TelegramAdapter.edit_message).parameters["parse_mode"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.asyncio
@pytest.mark.parametrize("finalize", [False, True])
async def test_parse_mode_content_is_sent_verbatim(finalize):
    adapter = _adapter()

    result = await adapter.edit_message("c1", "900", HTML, finalize=finalize, parse_mode="HTML")

    assert result.success is True and result.message_id == "900"
    adapter._bot.edit_message_text.assert_awaited_once()
    kwargs = _last_edit_kwargs(adapter)
    assert kwargs["text"] == HTML  # no MarkdownV2 conversion: it would escape the tags
    assert kwargs["parse_mode"] == "HTML"
    assert kwargs["message_id"] == 900


@pytest.mark.asyncio
async def test_parse_mode_final_edit_skips_the_rich_upgrade():
    adapter = _adapter()
    adapter._rich_send_disabled = False
    adapter._rich_eligible = lambda content: True
    adapter._try_edit_rich = AsyncMock(return_value=SendResult(success=True, message_id="900"))

    await adapter.edit_message("c1", "900", "| a | b |\n|---|---|\n| 1 | 2 |", finalize=True)
    adapter._try_edit_rich.assert_awaited_once()  # control: a Markdown final edit takes the rich path

    await adapter.edit_message("c1", "900", HTML, finalize=True, parse_mode="HTML")
    assert adapter._try_edit_rich.await_count == 1  # unchanged: caller-formatted content is not re-rendered
    adapter._bot.edit_message_text.assert_awaited_once()
    assert _last_edit_kwargs(adapter)["parse_mode"] == "HTML"


@pytest.mark.asyncio
async def test_parse_mode_edit_waits_for_a_busy_slot_instead_of_being_skipped(monkeypatch):
    """An interim plain edit is skipped while the shared send+edit slot is held (#116312); a formatted edit
    is a deliberate message nothing supersedes, so it waits for the slot like a send does."""
    adapter = _adapter()
    adapter._telegram_chat_outbound_slot_secs = 60.0  # held for the whole test however slow the runner is
    sleep = AsyncMock()
    monkeypatch.setattr("plugins.platforms.telegram.telegram_edit.asyncio.sleep", sleep)

    assert (await adapter.edit_message("c1", "900", "preview", finalize=False)).success is True
    assert adapter._chat_outbound_slot_remaining("c1") > 0
    skipped = await adapter.edit_message("c1", "900", "preview two", finalize=False)
    assert skipped.raw_response == {"skipped": True}  # control: the plain interim edit is skipped
    assert adapter._bot.edit_message_text.await_count == 1

    result = await adapter.edit_message("c1", "900", HTML, finalize=False, parse_mode="HTML")

    assert result.success is True
    assert not (result.raw_response or {}).get("skipped")
    sleep.assert_awaited_once()
    assert sleep.await_args.args[0] > 0  # waited out the held slot rather than firing into it
    assert adapter._bot.edit_message_text.await_count == 2
    assert _last_edit_kwargs(adapter)["parse_mode"] == "HTML"
    assert adapter._chat_outbound_slot_remaining("c1") > 0  # and re-armed the slot after firing


@pytest.mark.asyncio
async def test_long_markup_is_neither_preflight_truncated_nor_split():
    """Tags do not count toward Telegram's 4096 cap, so raw length is no reason to truncate or split."""
    adapter = _adapter()
    content = "<b>x</b>" * 1000  # 8000 code units raw, 1000 visible

    result = await adapter.edit_message("c1", "900", content, finalize=False, parse_mode="HTML")

    assert result.success is True
    assert _last_edit_kwargs(adapter)["text"] == content
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejected_markup_fails_instead_of_falling_back_to_plain():
    """The caller decides how to degrade (the plain form of its screen is not the raw tags), so the
    rejection must surface: the same error on a Markdown final edit is retried as plain text."""
    rejection = Exception("Bad Request: can't parse entities: Unsupported start tag \"x\" at byte offset 0")

    adapter = _adapter()
    adapter._bot.edit_message_text = AsyncMock(side_effect=rejection)
    control = await adapter.edit_message("c1", "900", "<x>oops</x>", finalize=True)
    assert control.success is False
    assert adapter._bot.edit_message_text.await_count == 2  # control: MarkdownV2, then the plain fallback

    adapter = _adapter()
    adapter._bot.edit_message_text = AsyncMock(side_effect=rejection)
    result = await adapter.edit_message("c1", "900", "<x>oops</x>", finalize=True, parse_mode="HTML")

    assert result.success is False
    assert "parse entities" in (result.error or "")
    adapter._bot.edit_message_text.assert_awaited_once()
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_too_long_markup_is_refused_not_split():
    adapter = _adapter()
    adapter._bot.edit_message_text = AsyncMock(side_effect=Exception("Bad Request: message is too long"))

    result = await adapter.edit_message("c1", "900", HTML, finalize=True, parse_mode="HTML")

    assert result.success is False
    assert "too long" in (result.error or "").lower()
    adapter._bot.edit_message_text.assert_awaited_once()  # no truncated retry
    adapter._bot.send_message.assert_not_awaited()  # no continuation messages


@pytest.mark.asyncio
async def test_flood_retry_keeps_parse_mode():
    class FloodError(Exception):
        retry_after = 0.01

    adapter = _adapter()
    adapter._bot.edit_message_text = AsyncMock(side_effect=[FloodError("Retry after 0"), MagicMock()])

    result = await adapter.edit_message("c1", "900", HTML, finalize=True, parse_mode="HTML")

    assert result.success is True
    assert adapter._bot.edit_message_text.await_count == 2
    assert _last_edit_kwargs(adapter)["text"] == HTML
    assert _last_edit_kwargs(adapter)["parse_mode"] == "HTML"


@pytest.mark.asyncio
async def test_without_parse_mode_an_interim_edit_is_still_skipped_on_a_busy_slot():
    """The legacy path is untouched: a plain interim edit is plain text and defers to the slot."""
    adapter = _adapter()
    adapter._telegram_chat_outbound_slot_secs = 60.0  # held for the whole test however slow the runner is

    assert (await adapter.edit_message("c1", "900", "preview", finalize=False)).success is True
    assert "parse_mode" not in _last_edit_kwargs(adapter)
    skipped = await adapter.edit_message("c1", "900", "preview two", finalize=False)

    assert skipped.success is True and skipped.raw_response == {"skipped": True}
    adapter._bot.edit_message_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_without_parse_mode_a_final_edit_is_still_converted_to_markdownv2():
    adapter = _adapter()

    result = await adapter.edit_message("c1", "900", MARKDOWN, finalize=True)

    assert result.success is True
    kwargs = _last_edit_kwargs(adapter)
    assert kwargs["parse_mode"] == ParseMode.MARKDOWN_V2
    assert kwargs["text"] == adapter.format_message(MARKDOWN) != MARKDOWN


@pytest.mark.asyncio
async def test_without_parse_mode_a_rejected_final_edit_still_falls_back_to_plain():
    adapter = _adapter()
    adapter._bot.edit_message_text = AsyncMock(
        side_effect=[Exception("Bad Request: can't parse entities"), MagicMock()])

    result = await adapter.edit_message("c1", "900", MARKDOWN, finalize=True)

    assert result.success is True
    assert adapter._bot.edit_message_text.await_count == 2
    kwargs = _last_edit_kwargs(adapter)
    assert "parse_mode" not in kwargs
    assert kwargs["text"] == _strip_mdv2(MARKDOWN)


@pytest.mark.asyncio
async def test_not_modified_with_parse_mode_is_a_successful_no_op():
    adapter = _adapter()
    adapter._bot.edit_message_text = AsyncMock(side_effect=Exception("Bad Request: message is not modified"))

    result = await adapter.edit_message("c1", "900", HTML, finalize=False, parse_mode="HTML")

    assert result.success is True and result.message_id == "900"
    adapter._bot.edit_message_text.assert_awaited_once()
