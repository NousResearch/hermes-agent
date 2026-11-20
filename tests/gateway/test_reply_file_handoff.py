"""A reply past the adapter's split cap (``MAX_SPLIT_MESSAGES``) arrives whole as one file (#152012).

A 415k-char reply once went out as 402 Telegram messages and flood-banned the bot. A reply that
needs more messages than the cap now goes out as one ``.md`` document holding the exact full text,
and the messages already posted for it are deleted once the file is confirmed. A refused upload
deletes nothing and the full text still arrives as messages. These drive the REAL consumer and
Telegram adapter against a fake Bot.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture(autouse=True)
def _parse_mode(monkeypatch):
    monkeypatch.setattr(telegram_adapter, "ParseMode", SimpleNamespace(MARKDOWN_V2="MarkdownV2"))


class _Bot:
    """Records every Bot API call; ``files`` holds each uploaded document's bytes."""

    def __init__(self, *, refuse_documents: bool = False):
        self.calls: list[tuple[str, str]] = []
        self.files: list[bytes] = []
        self.deleted: list[int] = []
        self._next = 1000
        self._refuse = refuse_documents

    def _id(self) -> int:
        self._next += 1
        return self._next

    async def send_message(self, text="", **_kw):
        self.calls.append(("send", text))
        return SimpleNamespace(message_id=self._id())

    async def edit_message_text(self, text="", **_kw):
        self.calls.append(("edit", text))
        return True

    async def send_document(self, document=None, **_kw):
        if self._refuse:
            raise RuntimeError("Bad Request: document upload refused")
        self.files.append(document.read())
        self.calls.append(("document", _kw.get("filename", "")))
        return SimpleNamespace(message_id=self._id())

    async def delete_message(self, message_id=None, **_kw):
        self.deleted.append(int(message_id))
        return True

    async def send_chat_action(self, **_kw):
        return True


def _telegram(**bot_kw) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._rich_send_disabled = True
    adapter._telegram_chat_outbound_slot_secs = 0.0
    adapter._bot = _Bot(**bot_kw)
    return adapter


def _body(chars: int) -> str:
    return "\n".join(" ".join(f"w{i}-{j}" for j in range(10)) for i in range(chars // 70))


async def _stream(adapter, text: str, *, step: int) -> GatewayStreamConsumer:
    consumer = GatewayStreamConsumer(adapter, "4242", config=StreamConsumerConfig(
        cursor=" ▉", edit_interval=0.01, buffer_threshold=10**9))
    task = asyncio.create_task(consumer.run())
    for pos in range(0, len(text), step):
        consumer.on_delta(text[pos:pos + step])
        await asyncio.sleep(0.003)  # health: allow HX013 -- the pace of the stream IS the behaviour (seals between deltas)
    consumer.finish(text)
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, 60)
    return consumer


def _kinds(adapter) -> list[str]:
    return [kind for kind, _ in adapter._bot.calls]


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [2_000, 10**9], ids=["slow-growing", "overflows-before-first-preview"])
async def test_an_over_cap_streamed_reply_arrives_as_one_file_and_its_messages_are_deleted(step):
    adapter = _telegram()
    text = _body(300_000)
    consumer = await _stream(adapter, text, step=step)

    bot = adapter._bot
    assert bot.files == [text.encode("utf-8")]  # the exact full reply, one file
    sent_ids = set(range(1001, 1001 + _kinds(adapter).count("send")))
    assert set(bot.deleted) == sent_ids  # every message this reply posted is gone
    assert _kinds(adapter).count("send") < adapter.MAX_SPLIT_MESSAGES
    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches(text) is True  # the gateway sends nothing more


@pytest.mark.asyncio
async def test_a_refused_file_deletes_nothing_and_the_full_text_arrives_as_messages():
    adapter = _telegram(refuse_documents=True)
    text = _body(60_000)
    consumer = await _stream(adapter, text, step=2_000)

    bot = adapter._bot
    assert bot.files == [] and bot.deleted == []
    shown = "".join(t for kind, t in bot.calls if kind in ("send", "edit")).replace("\\", "")
    assert all(line in shown for line in text.splitlines()[-30:])  # the tail is there too
    assert consumer.delivered_final_matches(text) is True


@pytest.mark.asyncio
async def test_a_reply_under_the_cap_is_unchanged():
    adapter = _telegram()
    text = _body(10_000)
    consumer = await _stream(adapter, text, step=500)

    assert "document" not in _kinds(adapter) and adapter._bot.deleted == []
    shown = "".join(t for _k, t in adapter._bot.calls).replace("\\", "")
    assert all(word in shown for word in text.split()[-20:])
    assert consumer.delivered_final_matches(text) is True


@pytest.mark.asyncio
async def test_a_non_streamed_reply_over_the_cap_is_only_the_file():
    adapter = _telegram()
    text = _body(100_000)
    result = await adapter._send_with_retry("4242", text)  # the gateway's final-reply send

    assert result.success and _kinds(adapter) == ["document"]
    assert adapter._bot.files == [text.encode("utf-8")]
