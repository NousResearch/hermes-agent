"""Telegram split-delivery cap (regression for #152012).

One degenerate 415,742-char final reply reached a Telegram DM as 402 back-to-back
``sendMessage`` calls and held the chat in flood control. Like Discord (#86581), a reply that
splits past ``MAX_SPLIT_MESSAGES`` keeps its first N-1 chunks and ends with one visible
truncation notice; a partial send never offers the dropped tail for resumption.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import utf16_len
from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter

MAX = TelegramAdapter.MAX_MESSAGE_LENGTH
MARKDOWN_V2 = "MarkdownV2"


@pytest.fixture(autouse=True)
def _parse_mode(monkeypatch):
    # The SDK is optional and absent here; MarkdownV2 sends need its ParseMode constant.
    monkeypatch.setattr(telegram_adapter, "ParseMode", SimpleNamespace(MARKDOWN_V2=MARKDOWN_V2))


class _FloodError(Exception):
    def __init__(self, seconds: float):
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


def _huge_content(chars: int = 150_000) -> str:
    # Distinct filler (size, not repetition) — about 35 Telegram-sized chunks.
    return "\n".join(" ".join(f"w{i}-{j}" for j in range(12)) for i in range(chars // 80))


def _uncapped_chunks(adapter: TelegramAdapter, content: str) -> list[str]:
    return adapter.truncate_message(adapter.format_message(content), MAX, len_fn=utf16_len)


def _content_splitting_into(adapter: TelegramAdapter, n: int) -> str:
    """The shortest ``_huge_content`` (whole 80-char lines) that ``send()`` splits into ``n`` chunks."""
    lo, hi = 1, n * MAX // 40  # line counts; ``hi`` lines are well past ``n`` chunks
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if len(_uncapped_chunks(adapter, _huge_content(mid * 80))) < n else (lo, mid)
    content = _huge_content(lo * 80)
    assert len(_uncapped_chunks(adapter, content)) == n
    return content


def _adapter(sent: list, *, fail_on_call: int | None = None) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._rich_send_disabled = True
    adapter._bot = MagicMock()

    async def fake_send_message(text: str, parse_mode=None, **_kw):
        if fail_on_call is not None and len(sent) + 1 == fail_on_call:
            raise _FloodError(120.0)
        sent.append((text, parse_mode))
        return MagicMock(message_id=1000 + len(sent))

    adapter._bot.send_message = AsyncMock(side_effect=fake_send_message)
    adapter._bot.edit_message_text = AsyncMock(return_value=True)
    return adapter


def _assert_notice(text: str, parse_mode) -> None:
    assert "Response truncated" in text
    assert parse_mode == MARKDOWN_V2
    assert "\\(" in text and utf16_len(text) <= MAX  # MarkdownV2-escaped, fits one message


@pytest.mark.asyncio
async def test_send_over_cap_delivers_at_most_cap_messages_ending_with_notice():
    sent: list = []
    adapter = _adapter(sent)
    content = _huge_content()
    cap = adapter.MAX_SPLIT_MESSAGES
    uncapped = adapter.truncate_message(adapter.format_message(content), MAX, len_fn=utf16_len)
    assert len(uncapped) > 3 * cap

    result = await adapter.send("4242", content)

    assert result.success is True
    assert len(sent) == cap
    assert [t.split()[0] for t, _ in sent[:-1]] == [c.split()[0] for c in uncapped[: cap - 1]]
    _assert_notice(*sent[-1])
    assert sent[0][0].endswith(f"\\(1/{cap}\\)")  # counters reflect what is actually sent
    assert len(result.raw_response["message_ids"]) == cap


@pytest.mark.asyncio
async def test_send_under_cap_is_untouched():
    sent: list = []
    adapter = _adapter(sent)
    content = _huge_content(9_000)
    uncapped = adapter.truncate_message(adapter.format_message(content), MAX, len_fn=utf16_len)
    assert 1 < len(uncapped) <= adapter.MAX_SPLIT_MESSAGES

    result = await adapter.send("4242", content)

    assert result.success is True
    assert len(sent) == len(uncapped)
    assert all("Response truncated" not in t for t, _ in sent)


@pytest.mark.asyncio
async def test_send_of_exactly_cap_chunks_is_untouched():
    sent: list = []
    adapter = _adapter(sent)
    cap = adapter.MAX_SPLIT_MESSAGES
    content = _content_splitting_into(adapter, cap)

    result = await adapter.send("4242", content)

    assert result.success is True
    assert len(sent) == cap
    assert all("Response truncated" not in t for t, _ in sent)
    assert sent[-1][0].endswith(f"\\({cap}/{cap}\\)")


@pytest.mark.asyncio
async def test_send_of_cap_plus_one_chunks_keeps_cap_minus_one_and_the_notice():
    sent: list = []
    adapter = _adapter(sent)
    cap = adapter.MAX_SPLIT_MESSAGES
    content = _content_splitting_into(adapter, cap + 1)
    uncapped = _uncapped_chunks(adapter, content)

    result = await adapter.send("4242", content)

    assert result.success is True
    assert len(sent) == cap
    assert [t.split()[0] for t, _ in sent[:-1]] == [c.split()[0] for c in uncapped[: cap - 1]]
    _assert_notice(*sent[-1])


@pytest.mark.asyncio
async def test_send_over_cap_without_warnings_drops_tail_silently(monkeypatch):
    sent: list = []
    adapter = _adapter(sent)
    monkeypatch.setattr(adapter, "warning_notifications_enabled", lambda *a, **k: False)

    result = await adapter.send("4242", _huge_content())

    assert result.success is True
    assert len(sent) == adapter.MAX_SPLIT_MESSAGES - 1
    assert all("Response truncated" not in t for t, _ in sent)


@pytest.mark.asyncio
async def test_partial_over_cap_send_never_offers_the_dropped_tail_for_resume():
    sent: list = []
    adapter = _adapter(sent, fail_on_call=2)
    cap = adapter.MAX_SPLIT_MESSAGES

    result = await adapter.send("4242", _huge_content())

    assert result.success is False and result.error.startswith("flood_control:")
    raw = result.raw_response
    assert raw["partial_overflow"] is True and raw["delivered_chunks"] == 1
    assert raw["total_chunks"] == cap and len(raw["undelivered_chunks"]) == cap - 1
    assert "Response truncated" in raw["undelivered_chunks"][-1]


@pytest.mark.asyncio
async def test_final_edit_overflow_is_capped():
    sent: list = []
    adapter = _adapter(sent)
    cap = adapter.MAX_SPLIT_MESSAGES

    result = await adapter.edit_message("4242", "201", _huge_content(), finalize=True)

    assert result.success is True
    assert adapter._bot.edit_message_text.await_count == 1
    assert len(sent) == cap - 1
    _assert_notice(*sent[-1])
    assert len(result.continuation_message_ids) == cap - 1



@pytest.mark.asyncio
async def test_a_capped_streamed_final_is_not_sent_again_by_the_gateway():
    """The gateway cancels a consumer still delivering a huge final after its flush window, and the
    cancel path's capped overflow edit IS the turn's delivery. It never matches the full final, and a
    mismatch made the stale-finalize path send the same capped chunks a second time (16, not 8)."""
    import asyncio
    import contextlib

    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    from tests.waiting import async_wait_until

    sent: list = []
    adapter = _adapter(sent)
    final = "<<head>> " + _huge_content()
    consumer = GatewayStreamConsumer(adapter, "4242", config=StreamConsumerConfig(
        cursor=" ▉", edit_interval=30.0, buffer_threshold=10**9))
    task = asyncio.create_task(consumer.run())
    consumer.on_delta(final[:200])
    await async_wait_until(lambda: sent, what="the first preview")
    # The consumer had drained only part of the final when the flush window ran out.
    consumer._accumulated = consumer._stream_ledger = final[:100_000]
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert len(sent) == adapter.MAX_SPLIT_MESSAGES  # the preview, then 6 continuations and the notice
    assert "Response truncated" in sent[-1][0]
    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches(final) is True
    assert consumer.delivered_final_matches("another answer " + final) is False
