"""A busy-path reply to an internal wake never hands the wake's producer identity to the adapter.

An admitted automation wake carries its producer identity (a JSON triple such as
``["kanban-wake", route, uuid]``) in ``MessageEvent.message_id``. ``_reply_anchor_for_event``
already refuses to anchor on it, but the busy path's ``_busy_reply_to`` fell back to the raw
``event.message_id`` outside Telegram threads, so a drain notice for a plugin wake reached
Telegram's ``int()`` as ``reply_to`` (#139611).
"""

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource

WAKE_ID = '["plugin","demo","agent:main:telegram:dm:42","0f1e2d3c4b5a69788796a5b4c3d2e1f0"]'


class _Adapter:
    def __init__(self):
        self.sends = []

    async def _send_with_retry(self, **kwargs):
        self.sends.append(kwargs)


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    return runner


@pytest.mark.parametrize("thread_id,chat_type", [(None, "dm"), ("7", "group"), ("7", "dm")])
@pytest.mark.asyncio
async def test_busy_reply_to_an_internal_wake_is_unanchored(thread_id, chat_type):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="u1",
                           chat_type=chat_type, thread_id=thread_id)
    event = MessageEvent(text="wake", source=source, message_id=WAKE_ID, internal=True,
                         allow_gateway_control=False)
    adapter = _Adapter()
    await _runner()._send_busy_reply(event, adapter, "draining")
    assert adapter.sends[0]["reply_to"] is None


@pytest.mark.asyncio
async def test_busy_reply_to_a_human_message_still_anchors():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="u1", chat_type="dm")
    event = MessageEvent(text="hi", source=source, message_id="1234")
    adapter = _Adapter()
    await _runner()._send_busy_reply(event, adapter, "draining")
    assert adapter.sends[0]["reply_to"] == "1234"
