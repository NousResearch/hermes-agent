"""Split Telegram replies under flood control: no duplicated head, no interleaving, no hammering.

A reply past ``MAX_MESSAGE_LENGTH`` goes out as several ``sendMessage`` calls. When chunk 2 is refused
(``RetryAfter`` past the inline cap) the adapter must report the partial delivery via the existing
``partial_overflow`` contract and ``_send_with_retry`` must resume from the undelivered remainder — never
re-send chunk 1 (the reporter saw a duplicated head), never drop the tail. A refusal the adapter hands to
the delivery ledger keeps that promise too: the ledger's later redelivery sends only the remainder.
"""
import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _FloodError(Exception):
    def __init__(self, seconds: float):
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


def _adapter(send_message: AsyncMock) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._rich_send_disabled = True
    adapter._bot = MagicMock()
    adapter._bot.send_message = send_message
    return adapter


def _three_chunk_text() -> str:
    return "\n".join(" ".join(f"w{i * 20 + j}" for j in range(20)) for i in range(80))  # ~9.5k chars, 3 chunks


@pytest.mark.asyncio
async def test_split_send_resumes_from_undelivered_tail_without_resending_head(monkeypatch):
    """Chunk 2 refused with a 7s RetryAfter (> the 5s adapter cap, < the 60s base cap): the retry
    delivers chunks 2..3 only — every chunk exactly once, in order."""
    sent: list = []
    calls = {"n": 0}

    async def fake_send_message(text: str, **_kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise _FloodError(7.0)
        sent.append(text)
        return MagicMock(message_id=1000 + calls["n"])

    adapter = _adapter(AsyncMock(side_effect=fake_send_message))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())

    async def _penalty_elapses(_delay):  # the base retry sleep is mocked; model the window having passed
        adapter._telegram_send_cooldown_until.clear()

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", _penalty_elapses)
    content = _three_chunk_text()
    expected = adapter.truncate_message(adapter.format_message(content), adapter.MAX_MESSAGE_LENGTH)
    assert len(expected) >= 3

    result = await adapter._send_with_retry(chat_id="4242", content=content)

    assert result.success is True
    assert [t.split()[0] for t in sent] == [c.split()[0] for c in expected]  # 1, 2, 3 — no duplicate head
    assert result.raw_response["message_ids"] == ["1001"] + [str(1002 + i) for i in range(1, len(expected))]


@pytest.mark.asyncio
async def test_over_cap_flood_returns_partial_overflow_and_arms_cooldown(monkeypatch):
    """A refusal past the base inline cap is returned typed (ledger owns the wait) but now carries the
    partial_overflow contract; the next send to that chat fails closed locally, another chat is unaffected."""
    calls = {"n": 0}

    async def fake_send_message(text: str, **_kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise _FloodError(120.0)
        return MagicMock(message_id=1000 + calls["n"])

    adapter = _adapter(AsyncMock(side_effect=fake_send_message))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())

    content = _three_chunk_text()
    total = len(adapter.truncate_message(adapter.format_message(content), adapter.MAX_MESSAGE_LENGTH))
    result = await adapter._send_with_retry(chat_id="4242", content=content)

    assert result.success is False and result.error == "flood_control:120.0"
    raw = result.raw_response
    assert raw["partial_overflow"] is True and raw["delivered_chunks"] == 1 and raw["total_chunks"] == total >= 3
    assert raw["last_message_id"] == "1001" and len(raw["undelivered_chunks"]) == total - 1
    assert calls["n"] == 2
    # Facet 3: a follow-up send inside the penalty window makes no API call; a different chat still sends.
    again = await adapter.send("4242", "hello again")
    assert again.success is False and again.error.startswith("flood_control:") and calls["n"] == 2
    other = await adapter.send("999", "hello other")
    assert other.success is True and calls["n"] == 3


@pytest.mark.asyncio
async def test_concurrent_split_sends_to_one_chat_do_not_interleave():
    """Two 3-chunk sends racing on one chat arrive as A A A B B B, not A B A B A B."""
    order: list = []

    async def fake_send_message(text: str, **_kw):
        await asyncio.sleep(0)  # yield like a real round-trip so an unlocked loop interleaves
        order.append(text.split()[0])
        return MagicMock(message_id=len(order))

    adapter = _adapter(AsyncMock(side_effect=fake_send_message))
    long = lambda tag: "\n".join(" ".join([tag] * 30) for _ in range(90))  # noqa: E731

    await asyncio.gather(adapter.send("1", long("REPORT")), adapter.send("1", long("ALERT")))

    assert len(order) >= 4
    switches = sum(1 for a, b in zip(order, order[1:]) if a != b)
    assert switches == 1, order


@pytest.mark.asyncio
async def test_ledger_redelivery_of_a_partial_final_sends_only_the_undelivered_tail(monkeypatch):
    """The over-cap refusal above leaves the final to the delivery ledger. Through the whole obligation path
    (``send_final_ledgered`` -> ledger row -> the runner's timed redelivery once the wait has passed), every
    word of the reply must reach the chat exactly once: the redelivery sends the tail, not the head again."""
    from gateway import delivery_ledger as dl
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    sent: list = []
    calls = {"n": 0}

    async def fake_send_message(text: str, **_kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise _FloodError(120.0)
        sent.append(text)
        return MagicMock(message_id=1000 + calls["n"])

    adapter = _adapter(AsyncMock(side_effect=fake_send_message))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner.session_store = None
    runner._async_session_store = MagicMock(_store=None, clear_resume_pending=AsyncMock())
    adapter.gateway_runner = runner
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm", user_id="u1")
    content = _three_chunk_text()

    result, _ = await adapter.send_final_ledgered(
        MessageEvent(text="write it up", source=source, message_id="m1"), "agent:main:telegram:dm:4242",
        content, {}, reply_to=None)
    assert result.success is False and result.error == "flood_control:120.0"

    # The platform's wait passes; the runner's flood timer then redelivers this bot's due rows.
    adapter._telegram_send_cooldown_until.clear()
    real_time = dl.time.time
    monkeypatch.setattr(dl, "time", MagicMock(time=lambda: real_time() + 200))
    assert await runner._redeliver_failed_obligations_for_platform(Platform.TELEGRAM) == 1

    delivered_words = [word for text in sent for word in re.findall(r"\bw\d+\b", text)]
    assert delivered_words == content.split()
