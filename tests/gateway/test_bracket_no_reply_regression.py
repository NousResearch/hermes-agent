from gateway.response_filters import is_intentional_silence_response, is_partial_silence_marker, is_autonomous_silence_response
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


@pytest.mark.asyncio
async def test_stream_consumer_never_sends_bracket_marker():
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id='1'))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id='1'))
    consumer = GatewayStreamConsumer(adapter, 'isolated', StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1))
    for part in ('[NO', '_RE', 'PLY]'):
        consumer.on_delta(part)
    consumer.finish()
    await consumer.run()
    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()
    assert not consumer.final_content_delivered


def test_bracket_no_reply_is_a_silence_marker():
    assert is_intentional_silence_response(' [NO_REPLY] ')


def test_bracket_no_reply_stream_prefix_is_held():
    assert is_partial_silence_marker('[NO_REP')


def test_prose_mention_is_not_silenced():
    assert not is_intentional_silence_response('What does [NO_REPLY] mean?')
    assert not is_autonomous_silence_response('[NO_REPLY] is the token being discussed')
