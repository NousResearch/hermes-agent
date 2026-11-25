"""Regression coverage for partial Telegram overflow delivery."""

import re
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _message(message_id: int | str) -> SimpleNamespace:
    return SimpleNamespace(message_id=message_id)


@pytest.fixture
def telegram_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    object.__setattr__(adapter, "MAX_MESSAGE_LENGTH", 160)
    return adapter


@pytest.mark.asyncio
async def test_edit_overflow_split_reports_later_partial_failure_after_some_continuations_land(telegram_adapter):
    """Partial metadata tracks the last delivered continuation before failure."""
    content = "word " * 120
    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)
    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[
            _message(202),
            RuntimeError("telegram send failed"),
            RuntimeError("telegram send failed"),
        ]
    )

    result = await telegram_adapter._edit_overflow_split(
        "12345", "201", content, finalize=False, metadata={"thread_id": "77"}
    )

    assert result.success is False
    assert result.message_id == "202"
    assert result.raw_response["partial_overflow"] is True
    assert result.raw_response["delivered_chunks"] == 2
    assert result.raw_response["last_message_id"] == "202"
    assert result.continuation_message_ids == ("202",)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcomes", [
    ["lost"],
    ["ok", "lost"],
    ["ok", "ok", "lost"],
    ["parse", "lost"],
    ["anchor", "lost"],
], ids=["first", "middle", "last", "plain_retry", "no_reply_retry"])
async def test_a_continuation_accepted_before_its_receipt_timed_out_is_not_resent(telegram_adapter, outcomes):
    """Regression for #160041: every row of the reply shows exactly once, and the turn stays uncertain."""
    import telegram.error as tg_error

    object.__setattr__(telegram_adapter, "MAX_MESSAGE_LENGTH", 600)
    content = "```text\n" + "\n".join(f"row {i}: some plain words" for i in range(80)) + "\n```"
    screen, pending = {"100": "preview"}, list(outcomes)

    async def edit_message_text(message_id=None, text="", **_kw):
        screen[str(message_id)] = text
        return True

    async def send_message(text="", **_kw):
        outcome = pending.pop(0) if pending else "ok"
        if outcome == "parse":
            raise tg_error.BadRequest("Can't parse entities")
        if outcome == "anchor":
            raise tg_error.BadRequest("Message to be replied not found")
        mid = 200 + len(screen)
        screen[str(mid)] = text
        if outcome == "lost":
            raise tg_error.TimedOut("Timed out")
        return _message(mid)

    telegram_adapter._bot.edit_message_text = AsyncMock(side_effect=edit_message_text)
    telegram_adapter._bot.send_message = AsyncMock(side_effect=send_message)
    consumer = GatewayStreamConsumer(telegram_adapter, "12345", StreamConsumerConfig())
    consumer._message_id, consumer._last_sent_text = "100", "preview"
    consumer._accumulated = consumer._stream_ledger = content

    if not await consumer._edit_existing(content, finalize=True, is_turn_final=True):
        await consumer._send_fallback_final(content)

    rows = Counter(re.findall(r"row \d+:", "\n".join(screen.values())))
    assert [row for row in re.findall(r"row \d+:", content) if rows[row] != 1] == []
    assert consumer.delivery_uncertain is True
    assert consumer.final_content_delivered is True
