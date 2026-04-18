import pytest

import gateway.platforms.feishu_streaming_card as feishu_streaming_card


class _FakeSession:
    def __init__(self, adapter, chat_id, metadata=None):
        self.adapter = adapter
        self.chat_id = chat_id
        self.metadata = metadata or {}
        self.updates = []
        self.close_calls = []
        self.card_id = "card_123"
        self.message_id = "msg_123"

    async def update(self, text: str) -> None:
        self.updates.append(text)

    async def close(self, *, final_text, model_label, elapsed_seconds, footer_text=None) -> None:
        self.close_calls.append(
            {
                "final_text": final_text,
                "model_label": model_label,
                "elapsed_seconds": elapsed_seconds,
                "footer_text": footer_text,
            }
        )


class _FailingCloseSession(_FakeSession):
    async def close(self, *, final_text, model_label, elapsed_seconds, footer_text=None) -> None:
        raise RuntimeError("close failed")


class _NoCardSession(_FakeSession):
    def __init__(self, adapter, chat_id, metadata=None):
        super().__init__(adapter, chat_id, metadata)
        self.card_id = None
        self.message_id = None


@pytest.mark.asyncio
async def test_feishu_card_stream_consumer_marks_final_response_sent(monkeypatch):
    monkeypatch.setattr(feishu_streaming_card, "_FeishuCardSession", _FakeSession)

    consumer = feishu_streaming_card.FeishuCardStreamConsumer(adapter=object(), chat_id="chat_123")
    consumer.on_delta("Hello")
    consumer.set_completion_meta(final_text="Hello world", model_label="gpt-test", elapsed_seconds=1.2)
    consumer.finish()

    await consumer.run()

    assert consumer.already_sent is True
    assert consumer.final_response_sent is True
    assert consumer._session.updates == ["Hello"]
    assert consumer._session.close_calls == [
        {
            "final_text": "Hello world",
            "model_label": "gpt-test",
            "elapsed_seconds": 1.2,
            "footer_text": None,
        }
    ]


@pytest.mark.asyncio
async def test_feishu_card_stream_consumer_does_not_mark_final_when_close_fails(monkeypatch):
    monkeypatch.setattr(feishu_streaming_card, "_FeishuCardSession", _FailingCloseSession)

    consumer = feishu_streaming_card.FeishuCardStreamConsumer(adapter=object(), chat_id="chat_123")
    consumer.on_delta("Hello")
    consumer.finish()

    await consumer.run()

    assert consumer.already_sent is True
    assert consumer.final_response_sent is False


@pytest.mark.asyncio
async def test_feishu_card_stream_consumer_does_not_claim_final_delivery_without_card(monkeypatch):
    monkeypatch.setattr(feishu_streaming_card, "_FeishuCardSession", _NoCardSession)

    consumer = feishu_streaming_card.FeishuCardStreamConsumer(adapter=object(), chat_id="chat_123")
    consumer.set_completion_meta(final_text="Hello world", model_label="gpt-test", elapsed_seconds=1.2)
    consumer.finish()

    await consumer.run()

    assert consumer.already_sent is False
    assert consumer.final_response_sent is False
