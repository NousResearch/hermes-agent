"""Telegram media uploads get budgets that cover the payload (#133093, #133164, #117795).

A Bot API server answers a media call only after the whole body reached Telegram and was
processed, so a fixed read/write budget cancels a large upload that still lands later. The
gateway's ``send_document`` and the standalone sender's media upload are driven with a recording
bot; the budgets that reach the Bot API call are asserted.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from hermes_agent.gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg
from plugins.platforms.telegram import standalone
from plugins.platforms.telegram import telegram_media_budget as budget
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Bot:
    def __init__(self, error: Exception | None = None):
        self.calls: list[dict] = []
        self.error = error

    async def _send(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(message_id=7)

    send_document = send_photo = send_video = _send


def _file(tmp_path, name: str, size: int) -> str:
    path = tmp_path / name
    with open(path, "wb") as f:
        f.truncate(size)
    return str(path)


def test_gateway_send_document_scales_budgets_with_file_size(tmp_path):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="x"))
    bot = adapter._bot = _Bot()
    deadlines: list = []
    real = tg._await_with_thread_deadline

    async def _spy(coro, *, timeout, **kw):
        deadlines.append(timeout)
        return await real(coro, timeout=timeout, **kw)

    tg._await_with_thread_deadline = _spy
    try:
        big = asyncio.run(adapter.send_document("123", _file(tmp_path, "big.pdf", 64 * 1024 * 1024)))
        small = asyncio.run(adapter.send_document("123", _file(tmp_path, "small.pdf", 0)))
    finally:
        tg._await_with_thread_deadline = real
    assert big.success and small.success
    transfer = 64 * 1024 * 1024 / budget.MIN_UPLOAD_RATE  # 512 s at the floor rate
    assert bot.calls[0]["read_timeout"] == tg._MEDIA_SEND_READ_TIMEOUT + transfer
    assert bot.calls[0]["write_timeout"] == budget.MEDIA_WRITE_TIMEOUT + transfer
    assert deadlines[0] == min(budget.DEADLINE_MAX, tg._MEDIA_SEND_DEADLINE + 1.5 * transfer)
    # An empty file keeps the fixed budgets: no write override, the fixed deadline.
    assert bot.calls[1]["read_timeout"] == tg._MEDIA_SEND_READ_TIMEOUT
    assert "write_timeout" not in bot.calls[1]
    assert deadlines[1] == tg._MEDIA_SEND_DEADLINE


def test_standalone_media_upload_uses_media_budgets(tmp_path):
    """The standalone upload (``hermes send``, cron, agent ``send_message``) used the Bot's general
    read timeout, which cuts off Telegram's post-upload processing of a delivered file."""
    bot = _Bot()
    path = _file(tmp_path, "book.pdf", 1024 * 1024)
    msg = asyncio.run(standalone._telegram_send_one_media(
        bot, 123, path, False, caption=None, parse_mode=None, has_html=False, thread_kwargs={}, force_document=False))
    assert msg.message_id == 7
    assert bot.calls[0]["read_timeout"] == budget.MEDIA_READ_TIMEOUT + 8.0  # 1 MiB at 128 KiB/s
    assert bot.calls[0]["write_timeout"] == budget.MEDIA_WRITE_TIMEOUT + 8.0
