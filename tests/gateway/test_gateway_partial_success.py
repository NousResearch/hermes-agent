import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource, build_session_key


class DummyAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []
        self.image_results = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def send_image(self, chat_id: str, image_url: str, caption=None, metadata=None, **kwargs):
        if self.image_results:
            return self.image_results.pop(0)
        return SendResult(success=True, message_id="img-1")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_event() -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="c1",
            user_id="u1",
            user_name="tester",
            chat_type="dm",
        ),
        message_id="m1",
    )


@pytest.mark.asyncio
async def test_process_message_background_sends_partial_failure_notice_when_text_succeeds_but_image_fails():
    adapter = DummyAdapter()
    adapter.image_results = [SendResult(success=False, error="boom")]

    async def handler(_event):
        return "Here you go https://example.com/test.png"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter._keep_typing = hold_typing
    adapter.extract_images = lambda response: ([('https://example.com/test.png', 'test image')], 'Here you go')

    event = _make_event()
    await adapter._process_message_background(event, build_session_key(event.source))

    assert len(adapter.sent) == 2
    assert adapter.sent[0]["content"] == "Here you go"
    assert "attachment" in adapter.sent[1]["content"].lower() or "image" in adapter.sent[1]["content"].lower()
