import pytest

from gateway.config import Platform, PlatformConfig
from gateway.final_sentinel import (
    FINAL_MESSAGE_SENTINEL,
    should_send_final_sentinel,
    strip_trailing_final_sentinel,
)
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource


class _SentinelAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="test"), platform)
        self.sent: list[tuple[str, str]] = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def get_chat_info(self, chat_id):
        return {}


def _event(platform: Platform, *, message_type: MessageType = MessageType.TEXT) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=message_type,
        source=SessionSource(platform=platform, chat_id="chat-1", chat_type="dm"),
    )


async def _run_background(adapter: _SentinelAdapter, event: MessageEvent, response: str) -> None:
    async def _handler(_event):
        return response

    adapter._message_handler = _handler
    await adapter._process_message_background(event, "telegram:chat-1")


@pytest.mark.asyncio
async def test_telegram_normal_response_sends_standalone_complete():
    adapter = _SentinelAdapter(Platform.TELEGRAM)

    await _run_background(adapter, _event(Platform.TELEGRAM), "main response")

    assert [content for _, content in adapter.sent] == ["main response", FINAL_MESSAGE_SENTINEL]


@pytest.mark.asyncio
async def test_discord_normal_response_sends_standalone_complete():
    adapter = _SentinelAdapter(Platform.DISCORD)

    await _run_background(adapter, _event(Platform.DISCORD), "main response")

    assert [content for _, content in adapter.sent] == ["main response", FINAL_MESSAGE_SENTINEL]


@pytest.mark.asyncio
async def test_non_target_platform_does_not_send_complete():
    adapter = _SentinelAdapter(Platform.SLACK)

    await _run_background(adapter, _event(Platform.SLACK), "main response")

    assert [content for _, content in adapter.sent] == ["main response"]


@pytest.mark.asyncio
async def test_command_response_does_not_send_complete():
    adapter = _SentinelAdapter(Platform.TELEGRAM)

    await _run_background(
        adapter,
        _event(Platform.TELEGRAM, message_type=MessageType.COMMAND),
        "system command response",
    )

    assert [content for _, content in adapter.sent] == ["system command response"]


def test_helper_strips_model_body_complete_line():
    assert strip_trailing_final_sentinel("Answer\n\nCOMPLETE") == "Answer"
    assert strip_trailing_final_sentinel("Answer\nCOMPLETE\n") == "Answer"
    assert strip_trailing_final_sentinel("Answer COMPLETE") == "Answer COMPLETE"


def test_helper_gates_failed_or_undelivered_turns():
    assert should_send_final_sentinel(
        platform=Platform.TELEGRAM,
        message_type=MessageType.TEXT,
        response_delivered=True,
    )
    assert not should_send_final_sentinel(
        platform=Platform.TELEGRAM,
        message_type=MessageType.TEXT,
        response_delivered=False,
    )
    assert not should_send_final_sentinel(
        platform=Platform.TELEGRAM,
        message_type=MessageType.TEXT,
        response_delivered=True,
        failed=True,
    )
