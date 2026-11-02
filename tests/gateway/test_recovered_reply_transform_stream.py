"""A transformed recovered final replaces its prior streamed segment, not its neighbors."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.test_stream_consumer_silence import _draft_adapter, _numbered_adapter, _visible


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["edit", "draft"])
@pytest.mark.parametrize("raw", ["Answer PRIVATE_VALUE", "PRIVATE_VALUE " * 200])
@pytest.mark.parametrize("final", ["Answer [REDACTED].", "Answer [REDACTED].\n\nFinal advisory."])
@pytest.mark.parametrize("fresh_final", [False, True])
async def test_changed_reply_replaces_only_its_prior_segment(transport, raw, final, fresh_final):
    adapter = _draft_adapter() if transport == "draft" else _numbered_adapter()
    ids = iter(range(1, 100))

    async def send(**kwargs):
        return SimpleNamespace(success=True, message_id=f"m{next(ids)}")

    adapter.MAX_MESSAGE_LENGTH = 1000
    adapter.send = AsyncMock(side_effect=send)
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    adapter.delete_message = AsyncMock(return_value=True)
    adapter.prefers_fresh_final_streaming = lambda *args, **kwargs: fresh_final
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(
        transport=transport, chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=1))
    consumer.on_delta("Earlier independent explanation.")
    consumer.on_delta(None)
    consumer.on_delta(raw)
    consumer.on_delta(None)
    consumer.on_delta("NO_REPLY")
    consumer.finish(final)
    await consumer.run()

    assert _visible(adapter) == ["Earlier independent explanation.", final]
    assert consumer.delivered_final_matches(final) is True
    assert all(call.args[1] != "m1" for call in adapter.delete_message.call_args_list)
    assert not consumer.has_durably_delivered_text(raw)
