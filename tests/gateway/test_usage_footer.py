import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.usage_footer import (
    append_usage_footer,
    clear_usage_footer_cache,
    get_usage_footer,
    maybe_append_usage_footer,
    send_usage_footer,
)


class _TelegramAdapter:
    platform = Platform.TELEGRAM
    name = "telegram"


class _DiscordAdapter:
    platform = Platform.DISCORD
    name = "discord"


class _BaseFooterAdapter(BasePlatformAdapter):
    name = "telegram"

    def __init__(self):
        super().__init__(PlatformConfig(), Platform.TELEGRAM)
        self.sent_contents = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent_contents.append(content)
        return SendResult(success=True, message_id=f"msg_{len(self.sent_contents)}")

    async def get_chat_info(self, chat_id):
        return {"name": "Test", "type": "dm"}


def test_get_usage_footer_returns_footer_for_telegram(monkeypatch):
    clear_usage_footer_cache()
    monkeypatch.setattr("gateway.usage_footer._load_usage_footer", lambda: "⚡ Test | 5h: 95% | Week: 96%")

    result = get_usage_footer(_TelegramAdapter(), "Hello world")

    assert result == "⚡ Test | 5h: 95% | Week: 96%"


def test_get_usage_footer_skips_non_telegram(monkeypatch):
    clear_usage_footer_cache()
    monkeypatch.setattr("gateway.usage_footer._load_usage_footer", lambda: "⚡ Test | 5h: 95% | Week: 96%")

    result = get_usage_footer(_DiscordAdapter(), "Hello world")

    assert result == ""


def test_get_usage_footer_dedupes_existing_footer(monkeypatch):
    clear_usage_footer_cache()
    footer = "⚡ Test | 5h: 95% | Week: 96%"
    monkeypatch.setattr("gateway.usage_footer._load_usage_footer", lambda: footer)

    result = get_usage_footer(_TelegramAdapter(), f"Hello world\n\n{footer}")

    assert result == ""


def test_get_usage_footer_recovers_after_transient_script_failure(monkeypatch):
    clear_usage_footer_cache()
    footer = "⚡ Test | 5h: 95% | Week: 96%"
    calls = {"count": 0}

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 30))
        return SimpleNamespace(stdout=f"{footer}\n")

    monkeypatch.setattr("gateway.usage_footer.subprocess.run", fake_run)

    assert get_usage_footer(_TelegramAdapter(), "Hello world") == ""
    assert get_usage_footer(_TelegramAdapter(), "Hello world") == footer
    assert calls["count"] == 2


def test_append_usage_footer_appends_once():
    footer = "⚡ Test | 5h: 95% | Week: 96%"

    assert append_usage_footer("Hello world", footer) == f"Hello world\n\n{footer}"
    assert append_usage_footer(f"Hello world\n\n{footer}", footer) == f"Hello world\n\n{footer}"


def test_maybe_append_usage_footer_skips_non_telegram(monkeypatch):
    clear_usage_footer_cache()
    monkeypatch.setattr("gateway.usage_footer._load_usage_footer", lambda: "⚡ Test | 5h: 95% | Week: 96%")

    assert maybe_append_usage_footer(_DiscordAdapter(), "Hello world") == "Hello world"


@pytest.mark.asyncio
async def test_base_send_with_retry_appends_footer_to_message(monkeypatch):
    clear_usage_footer_cache()
    footer = "⚡ Test | 5h: 95% | Week: 96%"
    monkeypatch.setattr("gateway.usage_footer._load_usage_footer", lambda: footer)
    adapter = _BaseFooterAdapter()

    result = await adapter._send_with_retry("chat_123", "Hello world", include_usage_footer=True)

    assert result.success
    assert adapter.sent_contents == [f"Hello world\n\n{footer}"]


@pytest.mark.asyncio
async def test_send_usage_footer_sends_separate_message(monkeypatch):
    adapter = MagicMock()
    adapter.platform = Platform.TELEGRAM
    adapter.name = "telegram"
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_2"))
    monkeypatch.setattr("gateway.usage_footer._load_usage_footer", lambda: "⚡ Test | 5h: 95% | Week: 96%")

    result = await send_usage_footer(adapter, "chat_123", "Hello world", {"thread_id": "99"})

    assert result.message_id == "msg_2"
    adapter.send.assert_awaited_once_with(
        chat_id="chat_123",
        content="⚡ Test | 5h: 95% | Week: 96%",
        metadata={"thread_id": "99"},
    )


def test_queued_first_response_resend_path_appends_footer():
    run_source = Path(__file__).parents[2].joinpath("gateway/run.py").read_text()

    assert "maybe_append_usage_footer" in run_source
    assert "first_response" in run_source
    assert "_status_thread_metadata" in run_source


@pytest.mark.asyncio
async def test_stream_consumer_final_send_edits_footer_into_final_message(monkeypatch):
    adapter = MagicMock()
    adapter.platform = Platform.TELEGRAM
    adapter.name = "telegram"
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))

    monkeypatch.setattr(
        "gateway.usage_footer._load_usage_footer",
        lambda: "⚡ Test | 5h: 95% | Week: 96%",
    )

    consumer = GatewayStreamConsumer(adapter, "chat_123", StreamConsumerConfig(cursor=""), metadata={"thread_id": "77"})
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Hello world")
    consumer.finish()
    await task

    sent_texts = [call.kwargs["content"] for call in adapter.send.await_args_list]
    assert sent_texts == ["Hello world"]
    adapter.edit_message.assert_awaited_once_with(
        chat_id="chat_123",
        message_id="msg_1",
        content="Hello world\n\n⚡ Test | 5h: 95% | Week: 96%",
        finalize=True,
    )


@pytest.mark.asyncio
async def test_stream_consumer_previewed_commentary_gets_standalone_footer(monkeypatch):
    adapter = MagicMock()
    adapter.platform = Platform.TELEGRAM
    adapter.name = "telegram"
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))

    monkeypatch.setattr(
        "gateway.usage_footer._load_usage_footer",
        lambda: "⚡ Test | 5h: 95% | Week: 96%",
    )

    consumer = GatewayStreamConsumer(adapter, "chat_123", StreamConsumerConfig(cursor=""), metadata={"thread_id": "77"})
    task = asyncio.create_task(consumer.run())
    consumer.on_commentary("Hello world")
    await asyncio.sleep(0.1)

    assert await consumer.finalize_previewed_response("Hello world") is True
    consumer.finish()
    await task

    sent_texts = [call.kwargs["content"] for call in adapter.send.await_args_list]
    assert sent_texts == ["Hello world", "⚡ Test | 5h: 95% | Week: 96%"]
    adapter.edit_message.assert_not_awaited()
    assert consumer.final_response_sent is True
