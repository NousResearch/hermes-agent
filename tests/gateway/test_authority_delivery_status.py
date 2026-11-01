"""Authority delivery keeps its verdict while retiring only the addressed typing status."""
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_ingress import deliver_response


class ThreadStatusAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=self.delivered, message_id='reply')

    async def stop_typing(self, chat_id, metadata=None):
        self.statuses.discard((chat_id, metadata['thread_id']))


@pytest.mark.asyncio
@pytest.mark.parametrize('response, success, expected', [
    (None, True, True),
    ('answer', True, True),
    ('answer', False, False),
])
async def test_delivery_verdict_and_thread_status_cleanup(response, success, expected):
    adapter = ThreadStatusAdapter(PlatformConfig(enabled=True), Platform.SLACK)
    adapter.delivered = success
    adapter.statuses = {('room', 'own-thread'), ('room', 'other-thread')}
    adapter.pause_typing_for_chat('room')
    event = MessageEvent(text='work', source=SessionSource(
        platform=Platform.SLACK, chat_id='room', chat_type='group', thread_id='own-thread'))
    # Transport result, not a verdict stub: exercise the real final-text delivery tracker.
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=success, message_id='reply'))

    assert await deliver_response(adapter, event, 'session', response) is expected
    assert adapter.statuses == {('room', 'other-thread')}
    assert 'room' in adapter._typing_paused
