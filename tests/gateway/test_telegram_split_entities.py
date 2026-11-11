"""Long Telegram replies keep MarkdownV2 inline entities intact across split chunks.

A split point chosen by newline/space can land inside ``*bold*``, ``_italic_``, ``~strike~`` or
``||spoiler||``. The chunks are judged by the Bot API stand-in's own MarkdownV2 check: a chunk it
refuses is what Telegram answers ``can't parse entities`` to (the user then gets stripped plain text).
"""

import re
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.fakes.platforms.telegram_standin import BotApiError, _check_text

UNITS = (
    "Mening med **viktig fetstil** mitt i texten som upprepas. ",
    "Some words with ~~struck through text~~ in the text again. ",
    "Some words with ||hidden spoiler words|| in the text again. ",
    "Words with *an italic stretch here* again and again. ",
)


class _BotApi:
    """Records each text call, refusing what the real Bot API refuses (via the stand-in's check)."""

    def __init__(self):
        self.calls: list[tuple[str | None, str]] = []

    async def _call(self, text="", parse_mode=None, **_kw):
        mode = "MarkdownV2" if parse_mode else None
        try:
            _check_text({"text": text, "parse_mode": mode})
        except BotApiError as exc:
            self.calls.append(("refused", text))
            raise RuntimeError(str(exc)) from None
        self.calls.append((mode, text))
        return MagicMock(message_id=100 + len(self.calls))

    async def send_message(self, chat_id=None, **kw):
        return await self._call(**kw)

    async def edit_message_text(self, chat_id=None, message_id=None, **kw):
        return await self._call(**kw)


def _adapter(bot: _BotApi) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:abc"))
    adapter._rich_send_disabled = True
    adapter._bot = bot
    return adapter


def _styled_everywhere(calls) -> bool:
    """Every chunk was accepted as MarkdownV2 and none shows an escaped (literal) style marker."""
    return all(mode == "MarkdownV2" and not re.search(r"\\[*~|]", text) for mode, text in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", UNITS)
async def test_send_split_never_breaks_an_inline_entity(unit):
    for pad in range(0, 60, 3):
        bot = _BotApi()
        result = await _adapter(bot).send("4242", "x" * pad + unit * (4400 // len(unit) + 2))
        assert result.success
        assert len(bot.calls) == 2 and _styled_everywhere(bot.calls), (pad, [text[-40:] for _, text in bot.calls])


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", UNITS)
async def test_overflow_edit_split_never_breaks_an_inline_entity(unit):
    for pad in range(0, 60, 3):
        bot = _BotApi()
        content = "x" * pad + unit * (4400 // len(unit) + 2)
        result = await _adapter(bot)._edit_overflow_split("4242", "99", content, finalize=True)
        assert result.success
        assert len(bot.calls) == 2 and _styled_everywhere(bot.calls), (pad, [text[-40:] for _, text in bot.calls])


@pytest.mark.asyncio
async def test_overflow_edit_split_leaves_literal_markers_and_a_final_backslash_alone():
    """An escaped ``\\*args`` is text, not an entity to carry; a reply ending in ``\\`` keeps it."""
    bot = _BotApi()
    content = "Use \\*args here. " + "plain words in the reply " * 220 + "C:\\"
    result = await _adapter(bot)._edit_overflow_split("4242", "99", content, finalize=True)
    assert result.success and len(bot.calls) == 2
    shown = "".join(re.sub(r" \(\d+/\d+\)$", "", _strip(text)) for _, text in bot.calls)
    assert shown.count("*") == 1 and shown.endswith("C:\\"), shown[-30:]


def _strip(text: str) -> str:
    return re.sub(r"\\(.)", r"\1", text)


def test_standalone_send_split_never_breaks_an_inline_entity(monkeypatch):
    """The out-of-gateway sender (cron delivery, send_message tool) splits like the live adapter."""
    import asyncio
    import sys
    from types import SimpleNamespace

    from plugins.platforms.telegram.standalone import _send_telegram

    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    for unit in UNITS:
        for pad in range(0, 60, 6):
            bot = _BotApi()
            constants = SimpleNamespace(ParseMode=parse_mode)
            monkeypatch.setitem(sys.modules, "telegram", SimpleNamespace(Bot=lambda *a, **k: bot, constants=constants))
            monkeypatch.setitem(sys.modules, "telegram.constants", constants)
            result = asyncio.run(_send_telegram("tok", "123", "x" * pad + unit * (4400 // len(unit) + 2)))
            assert result["success"] is True
            assert len(bot.calls) == 2 and _styled_everywhere(bot.calls), (unit, pad, [text[-40:] for _, text in bot.calls])
