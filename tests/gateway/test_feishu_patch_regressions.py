import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def test_interactive_card_keeps_first_100_lines():
    from plugins.platforms.feishu.adapter import normalize_feishu_message

    elements = [
        {"tag": "div", "text": {"tag": "plain_text", "content": f"line {i}"}}
        for i in range(120)
    ]
    normalized = normalize_feishu_message(
        message_type="interactive",
        raw_content=json.dumps({"card": {"elements": elements}}),
    )

    lines = normalized.text_content.splitlines()
    assert len(lines) == 100
    assert lines[-1] == "line 99"


def test_get_message_request_asks_for_user_card_content():
    from plugins.platforms.feishu.adapter import FeishuAdapter
    import plugins.platforms.feishu.adapter as feishu_adapter

    request = Mock()
    builder = Mock()
    builder.message_id.return_value = builder
    builder.build.return_value = request
    fake_request_cls = SimpleNamespace(builder=Mock(return_value=builder))

    with patch.object(feishu_adapter, "GetMessageRequest", fake_request_cls):
        result = FeishuAdapter._build_get_message_request("om_msg")

    assert result is request
    request.add_query.assert_called_once_with("card_msg_content_type", "user_card_content")


@pytest.mark.asyncio
async def test_feishu_thread_metadata_does_not_reply_in_thread():
    from gateway.config import PlatformConfig
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    captured = {}

    class ReplyAPI:
        def reply(self, request):
            captured["request"] = request
            return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_reply"))

    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=ReplyAPI())))

    async def direct(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=direct):
        result = await adapter.send(
            chat_id="oc_chat",
            content="hello",
            reply_to="om_parent",
            metadata={"thread_id": "omt-thread"},
        )

    assert result.success
    assert captured["request"].request_body.reply_in_thread is False


@pytest.mark.asyncio
async def test_reply_snippet_allows_3000_chars_with_chinese_marker():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = None
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_name="DM",
        chat_type="private",
        user_name="Alice",
    )
    event = MessageEvent(
        text="follow-up",
        source=source,
        reply_to_message_id="42",
        reply_to_text="x" * 3200,
    )

    result = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    assert result is not None
    assert result.startswith('[Replying to: "' + "x" * 3000 + '\n…[已截断，原文过长]"]')
    assert "x" * 3001 not in result
