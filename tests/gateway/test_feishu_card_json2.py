"""Tests for Feishu JSON 2.0 card sentinel delivery."""

from __future__ import annotations

import asyncio
import json

import pytest

from gateway.platforms.base import SendResult


_MINIMAL_CARD = {
    "schema": "2.0",
    "config": {"update_multi": True, "width_mode": "fill"},
    "header": {
        "title": {"tag": "plain_text", "content": "🚀 AI 潮流玩法雷达"},
        "template": "turquoise",
    },
    "body": {
        "elements": [
            {
                "tag": "markdown",
                "element_id": "m_top",
                "content": "30秒结论\n- Hermes no_agent cron 值得看",
            }
        ]
    },
}


def _sentinel_payload(card: dict | None = None) -> str:
    return "FEISHU_CARD_JSON_2_0\n" + json.dumps(card or _MINIMAL_CARD, ensure_ascii=False)


def test_feishu_card_json2_sentinel_builds_interactive_payload():
    from gateway.platforms.feishu import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)

    msg_type, payload = adapter._build_outbound_payload(_sentinel_payload())

    assert msg_type == "interactive"
    assert json.loads(payload) == _MINIMAL_CARD


@pytest.mark.parametrize(
    "content",
    [
        "```\n" + _sentinel_payload() + "\n```",
        "```json\n" + _sentinel_payload() + "\n```",
        "FEISHU_CARD_JSON_2_0\n```json\n" + json.dumps(_MINIMAL_CARD, ensure_ascii=False) + "\n```",
    ],
)
def test_feishu_card_json2_sentinel_accepts_wrapping_code_fences(content):
    from gateway.platforms.feishu import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)

    msg_type, payload = adapter._build_outbound_payload(content)

    assert msg_type == "interactive"
    assert json.loads(payload) == _MINIMAL_CARD


@pytest.mark.parametrize(
    "content",
    [
        "FEISHU_CARD_JSON_2_0\nnot-json",
        "FEISHU_CARD_JSON_2_0\n[]",
        _sentinel_payload({"schema": "1.0", "elements": []}),
        "Here is a card:\n" + _sentinel_payload(),
    ],
)
def test_feishu_card_json2_invalid_or_prefaced_content_is_not_interactive(content):
    from gateway.platforms.feishu import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)

    msg_type, _payload = adapter._build_outbound_payload(content)

    assert msg_type != "interactive"


@pytest.mark.asyncio
async def test_feishu_card_json2_send_does_not_split_large_card():
    from gateway.platforms.feishu import FeishuAdapter

    class _Adapter(FeishuAdapter):
        def __init__(self):
            self._client = object()
            self.calls = []

        async def _feishu_send_with_retry(self, **kwargs):
            self.calls.append(kwargs)
            return type("Response", (), {
                "success": lambda self: True,
                "data": type("Data", (), {"message_id": "om_card"})(),
            })()

    large_card = dict(_MINIMAL_CARD)
    large_card["body"] = {
        "elements": [
            {
                "tag": "markdown",
                "element_id": "m_big",
                "content": "x" * 9000,
            }
        ]
    }
    adapter = _Adapter()

    result = await adapter.send("oc_chat", _sentinel_payload(large_card))

    assert result.success is True
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["msg_type"] == "interactive"
    assert json.loads(adapter.calls[0]["payload"])["body"]["elements"][0]["content"] == "x" * 9000


class _FakeFeishuAdapter:
    MAX_MESSAGE_LENGTH = 4096
    SUPPORTS_MESSAGE_EDITING = True

    def __init__(self):
        self.sent: list[str] = []
        self.edited: list[str] = []

    def should_buffer_stream_update(self, content: str) -> bool:
        from gateway.platforms.feishu import FeishuAdapter

        return FeishuAdapter.should_buffer_stream_update(self, content)

    def truncate_message(self, text: str, _limit: int) -> list[str]:
        return [text]

    async def send(self, chat_id: str, content: str, **_kwargs) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id=f"om_{len(self.sent)}")

    async def edit_message(self, chat_id: str, message_id: str, content: str, **_kwargs) -> SendResult:
        self.edited.append(content)
        return SendResult(success=True, message_id=message_id)


@pytest.mark.parametrize(
    "deltas, expected_final",
    [
        (["FEISHU_CARD_JSON_2_0\n", json.dumps(_MINIMAL_CARD, ensure_ascii=False)], _sentinel_payload()),
        (["```json\nFEISHU_CARD_JSON_2_0\n", json.dumps(_MINIMAL_CARD, ensure_ascii=False), "\n```"], "```json\n" + _sentinel_payload() + "\n```"),
    ],
)
@pytest.mark.asyncio
async def test_stream_consumer_buffers_feishu_card_json2_until_done(deltas, expected_final):
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    adapter = _FakeFeishuAdapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "oc_chat",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=" ▉"),
    )
    task = asyncio.create_task(consumer.run())

    for delta in deltas:
        consumer.on_delta(delta)
        await asyncio.sleep(0.08)
        assert adapter.sent == []
        assert adapter.edited == []

    consumer.finish()
    await asyncio.wait_for(task, timeout=1)

    assert adapter.sent == [expected_final]
    assert adapter.edited == []
    assert consumer.final_response_sent is True
