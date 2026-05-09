"""Feishu stream + footer regression tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.feishu import FeishuAdapter
from gateway.runtime_footer import build_footer_line
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class FakeFeishuAdapter:
    MAX_MESSAGE_LENGTH = 8000
    SUPPORTS_MESSAGE_EDITING = True
    _render_mode = "card"
    _always_card = True

    @property
    def REQUIRES_EDIT_FINALIZE(self):
        return self._render_mode == "card" or self._always_card

    def __init__(self):
        self.sent = []
        self.edits = []
        self.recorded_progress = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="msg_1")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "finalize": finalize,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def record_execution_progress(self, chat_id, line, metadata=None):
        self.recorded_progress.append(
            {
                "chat_id": chat_id,
                "line": line,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="msg_1")

    def truncate_message(self, content, max_length):
        return [content]


class FakePreservedFeishuAdapter(FakeFeishuAdapter):
    PRESERVE_STREAM_TARGET_ACROSS_SEGMENTS = True


class FakeRunner:
    def __init__(self, adapter):
        self.adapters = {Platform.FEISHU: adapter}
        self.media_deliveries = []

    async def _deliver_media_from_response(self, response, event, adapter):
        self.media_deliveries.append((response, event, adapter))


@pytest.mark.asyncio
async def test_preserved_card_stream_records_segment_progress_before_reset():
    adapter = FakePreservedFeishuAdapter()
    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat_id",
        config=StreamConsumerConfig(edit_interval=999, buffer_threshold=999, cursor=""),
        metadata={"model": "provider/model-x"},
    )

    task = asyncio.create_task(consumer.run())
    consumer.on_delta("第一段进度")
    consumer.on_segment_break()
    consumer.on_delta("第二段进度")
    consumer.finish()
    await asyncio.wait_for(task, timeout=2)

    assert adapter.sent == [
        {
            "chat_id": "chat_id",
            "content": "第一段进度",
            "reply_to": None,
            "metadata": {"model": "provider/model-x", "_hermes_stream_preview": True},
        }
    ]
    assert adapter.recorded_progress == [
        {
            "chat_id": "chat_id",
            "line": "第一段进度",
            "metadata": {"model": "provider/model-x"},
        }
    ]
    assert adapter.edits[-1] == {
        "chat_id": "chat_id",
        "message_id": "msg_1",
        "content": "第二段进度",
        "finalize": True,
    }


@pytest.mark.asyncio
async def test_feishu_recorded_segment_progress_is_composed_into_next_card_edit():
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._streaming_card_state = {"card_123": {"started_at": 100.0}}
    adapter._execution_progress_lines = {}
    adapter._tool_progress_lines = {}
    adapter._latest_card_content = {"card_123": "第一段进度"}
    adapter._active_card_for_chat = {"chat_id": "card_123"}
    adapter.format_message = lambda content: content
    update_content = MagicMock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace()))
    adapter._client = SimpleNamespace(
        cardkit=SimpleNamespace(
            v1=SimpleNamespace(
                card_element=SimpleNamespace(content=update_content),
            )
        )
    )

    record_result = await adapter.record_execution_progress("chat_id", "第一段进度")
    edit_result = await adapter.edit_message("chat_id", "card_123", "第二段进度")

    assert record_result.success is True
    assert edit_result.success is True
    content_request = update_content.call_args_list[0].args[0]
    assert content_request.element_id == "content"
    assert content_request.request_body.content == "第一段进度\n...\n第二段进度"


@pytest.mark.asyncio
async def test_feishu_text_mode_does_not_force_redundant_finalize_edit():
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._render_mode = "text"
    adapter._always_card = False

    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat_id",
        config=StreamConsumerConfig(edit_interval=999, buffer_threshold=999, cursor=""),
    )

    assert consumer._adapter_requires_finalize is False


@pytest.mark.asyncio
async def test_feishu_card_mode_forces_finalize_edit_to_close_stream_card():
    adapter = FakeFeishuAdapter()
    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat_id",
        config=StreamConsumerConfig(edit_interval=999, buffer_threshold=999, cursor=""),
    )

    task = asyncio.create_task(consumer.run())
    consumer.on_delta("流式正文")
    consumer.finish()
    await asyncio.wait_for(task, timeout=2)

    assert consumer.final_response_sent is True
    assert adapter.sent == [
        {
            "chat_id": "chat_id",
            "content": "流式正文",
            "reply_to": None,
            "metadata": {"_hermes_stream_preview": True},
        }
    ]
    assert adapter.edits[-1] == {
        "chat_id": "chat_id",
        "message_id": "msg_1",
        "content": "流式正文",
        "finalize": True,
    }


def test_feishu_card_payload_has_streaming_config_and_footer_elements():
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True

    msg_type, payload = adapter._build_outbound_payload(
        "测试 stream footer",
        metadata={"_hermes_stream_preview": True},
    )
    card = __import__("json").loads(payload)

    assert msg_type == "interactive"
    assert card["schema"] == "2.0"
    assert card["config"]["streaming_mode"] is True
    assert card["config"]["streaming_config"]["print_frequency_ms"]["default"] == 50
    assert card["body"]["elements"][0]["tag"] == "markdown"
    assert card["body"]["elements"][0]["element_id"] == "content"
    footer_text = card["body"]["elements"][1]["content"]
    assert footer_text.startswith("Agent: main |")
    assert "Elapsed: 0s" in footer_text
    assert "生成中" in footer_text
    assert "生成耗时实时更新" not in footer_text


@pytest.mark.asyncio
async def test_feishu_card_send_uses_cardkit_card_id_as_edit_target():
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._streaming_card_state = {}
    adapter._create_card_message = AsyncMock(return_value="card_123")
    adapter._build_create_message_body = lambda **kwargs: SimpleNamespace(**kwargs)
    adapter._build_create_message_request = lambda receive_id_type, request_body: SimpleNamespace(
        receive_id_type=receive_id_type,
        request_body=request_body,
    )
    create = MagicMock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_message")))
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(create=create))))

    response = await adapter._send_raw_message(
        chat_id="chat_id",
        msg_type="interactive",
        payload='{"schema":"2.0"}',
        reply_to=None,
        metadata=None,
    )

    adapter._create_card_message.assert_awaited_once_with('{"schema":"2.0"}')
    request = create.call_args.args[0]
    body = request.request_body
    assert body.msg_type == "interactive"
    assert body.content == '{"type": "card", "data": {"card_id": "card_123"}}'
    assert response.hermes_feishu_card_id == "card_123"
    result = adapter._finalize_send_result(response, "send failed")
    assert result.message_id == "card_123"


@pytest.mark.asyncio
async def test_feishu_card_send_records_stream_start_time(monkeypatch):
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._client = object()
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._streaming_card_state = {}
    adapter.format_message = lambda content: content
    adapter.truncate_message = lambda content, max_length: [content]
    monkeypatch.setattr("gateway.platforms.feishu.time.monotonic", lambda: 123.45)
    response = SimpleNamespace(success=lambda: True, hermes_feishu_card_id="card_123")
    adapter._feishu_send_with_retry = AsyncMock(return_value=response)

    result = await adapter.send("chat_id", "流式正文", metadata={"_hermes_stream_preview": True})

    assert result.success is True
    assert result.message_id == "card_123"
    sent_payload = adapter._feishu_send_with_retry.await_args.kwargs["payload"]
    sent_card = __import__("json").loads(sent_payload)
    footer_text = sent_card["body"]["elements"][1]["content"]
    assert footer_text.startswith("Agent: main |")
    assert "Elapsed: 0s" in footer_text
    assert "生成中" in footer_text
    assert "生成耗时实时更新" not in footer_text
    assert adapter._streaming_card_state == {"card_123": {"started_at": 123.45}}


@pytest.mark.asyncio
async def test_feishu_plain_final_card_send_is_completed_not_streaming(monkeypatch):
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._client = object()
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._streaming_card_state = {}
    adapter._active_card_for_chat = {}
    adapter.format_message = lambda content: content
    adapter.truncate_message = lambda content, max_length: [content]
    monkeypatch.setattr("gateway.platforms.feishu.time.monotonic", lambda: 123.45)
    response = SimpleNamespace(success=lambda: True, hermes_feishu_card_id="card_final_1")
    adapter._feishu_send_with_retry = AsyncMock(return_value=response)

    result = await adapter.send(
        "chat_id",
        "最终正文",
        metadata={"model": "provider/model-x", "provider": "test"},
    )

    assert result.success is True
    assert result.message_id == "card_final_1"
    sent_payload = adapter._feishu_send_with_retry.await_args.kwargs["payload"]
    sent_card = __import__("json").loads(sent_payload)
    assert sent_card["config"]["streaming_mode"] is False
    footer_text = sent_card["body"]["elements"][1]["content"]
    assert "Model: model-x" in footer_text
    assert "Provider: test" in footer_text
    assert "Elapsed: 0s" in footer_text
    assert "✅ 已完成" in footer_text
    assert "生成中" not in footer_text
    assert adapter._streaming_card_state == {}
    assert adapter._active_card_for_chat == {}


@pytest.mark.asyncio
async def test_feishu_card_embeds_tool_progress_block_inline_in_same_card():
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._streaming_card_state = {}
    adapter._execution_progress_lines = {}
    adapter._tool_progress_lines = {}
    adapter._latest_card_content = {}
    adapter._active_card_for_chat = {}
    adapter._tool_progress_interval = 0
    adapter.format_message = lambda content: content

    update_content = MagicMock(
        return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace())
    )
    update_card = MagicMock(
        return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace())
    )
    settings_card = MagicMock(
        return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace())
    )
    update_message = MagicMock()
    adapter._client = SimpleNamespace(
        cardkit=SimpleNamespace(
            v1=SimpleNamespace(
                card_element=SimpleNamespace(content=update_content),
                card=SimpleNamespace(update=update_card, settings=settings_card),
            )
        ),
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(update=update_message))),
    )

    chat_id = "oc_chat_inline"
    # First tool fires before any answer text exists. In CardKit mode this now
    # opens the same streaming card immediately, instead of silently buffering
    # until final text arrives.
    msg_id = "card_inline_1"
    adapter._feishu_send_with_retry = AsyncMock(
        return_value=SimpleNamespace(success=lambda: True, hermes_feishu_card_id=msg_id)
    )
    await adapter.update_tool_progress(
        chat_id,
        ['💻 terminal: "first command"'],
        metadata={"model": "provider/model-x", "provider": "test"},
    )
    update_content.assert_not_called()
    update_card.assert_not_called()
    update_message.assert_not_called()
    assert adapter._active_card_for_chat[chat_id] == msg_id
    assert msg_id in adapter._streaming_card_state
    sent_payload = adapter._feishu_send_with_retry.await_args.kwargs["payload"]
    assert "collapsible_panel" in sent_payload
    assert "first command" in sent_payload
    assert "Model: model-x" in sent_payload
    assert "Provider: test" in sent_payload
    adapter._latest_card_content[msg_id] = "正文开始了"

    # First answer streaming chunk lands. The card already contains the progress
    # panel, so streaming text updates only the content/footer elements.
    await adapter.edit_message(chat_id, msg_id, "正文开始了")
    update_card.assert_not_called()
    update_message.assert_not_called()
    assert update_content.call_args_list[0].args[0].element_id == "content"
    assert update_content.call_args_list[0].args[0].request_body.content == "正文开始了"
    assert update_content.call_args_list[1].args[0].element_id == "footer"

    # Second tool fires: update_tool_progress edits the active CardKit card so
    # long tool-heavy turns remain visibly alive before any final answer text.
    update_card.reset_mock()
    update_content.reset_mock()
    await adapter.update_tool_progress(
        chat_id,
        ['💻 terminal: "first command"', '💻 terminal: "second command"'],
    )
    update_card.assert_called_once()
    update_content.assert_not_called()
    update_message.assert_not_called()
    assert len(adapter._tool_progress_lines.get(chat_id, [])) == 2

    # Next stream consumer edit updates the existing content/footer elements while
    # the progress panel stays in the card.
    update_card.reset_mock()
    update_content.reset_mock()
    await adapter.edit_message(chat_id, msg_id, "正文开始了更多")
    assert update_content.call_args_list[0].args[0].element_id == "content"
    assert update_content.call_args_list[0].args[0].request_body.content == "正文开始了更多"
    assert update_content.call_args_list[1].args[0].element_id == "footer"
    update_card.assert_not_called()

    # Finalize the card: tool block clears so the next reply starts clean.
    update_card.reset_mock()
    await adapter.edit_message(chat_id, msg_id, "正文开始了。最终答复", finalize=True)
    update_card.assert_called_once()
    assert chat_id not in adapter._tool_progress_lines
    assert chat_id not in adapter._active_card_for_chat


@pytest.mark.asyncio
async def test_feishu_send_reuses_progress_opened_card_for_first_answer_chunk():
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._client = object()
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._streaming_card_state = {"card_progress_1": {"started_at": 100.0}}
    adapter._tool_progress_lines = {"chat_id": ['💻 terminal: "scan"']}
    adapter._latest_card_content = {"card_progress_1": " "}
    adapter._active_card_for_chat = {"chat_id": "card_progress_1"}
    adapter.format_message = lambda content: content
    adapter.truncate_message = lambda content, max_length: [content]
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="card_progress_1"))
    adapter._feishu_send_with_retry = AsyncMock()

    result = await adapter.send("chat_id", "最终答复开始", metadata={"model": "provider/model-x"})

    assert result.success is True
    assert result.message_id == "card_progress_1"
    adapter.edit_message.assert_awaited_once_with(
        "chat_id",
        "card_progress_1",
        "最终答复开始",
        finalize=True,
    )
    adapter._feishu_send_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_feishu_tool_progress_updates_are_throttled_after_initial_card(monkeypatch):
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._tool_progress_interval = 1.5
    adapter._streaming_card_state = {"card_progress_1": {"started_at": 100.0}}
    adapter._tool_progress_lines = {}
    adapter._latest_card_content = {"card_progress_1": " "}
    adapter._active_card_for_chat = {"chat_id": "card_progress_1"}
    adapter._tool_progress_last_update = {"chat_id": 10.0}
    adapter._client = SimpleNamespace(
        cardkit=SimpleNamespace(
            v1=SimpleNamespace(
                card=SimpleNamespace(update=MagicMock(return_value=SimpleNamespace(success=lambda: True)))
            )
        )
    )
    monkeypatch.setattr("gateway.platforms.feishu.time.monotonic", lambda: 10.5)

    await adapter.update_tool_progress("chat_id", ["first", "second"])

    adapter._client.cardkit.v1.card.update.assert_not_called()
    assert adapter._tool_progress_lines["chat_id"] == ["first", "second"]


@pytest.mark.asyncio
async def test_feishu_edit_streaming_updates_card_content_element_not_whole_card():
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._streaming_card_state = {"card_123": {"started_at": 100.0}}
    adapter.format_message = lambda content: content
    update_content = MagicMock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace()))
    update_card = MagicMock()
    update_message = MagicMock()
    adapter._client = SimpleNamespace(
        cardkit=SimpleNamespace(
            v1=SimpleNamespace(
                card_element=SimpleNamespace(content=update_content),
                card=SimpleNamespace(update=update_card),
            )
        ),
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(update=update_message))),
    )

    result = await adapter.edit_message("chat_id", "card_123", "正在逐步输出", finalize=False)

    assert result.success is True
    assert result.message_id == "card_123"
    update_message.assert_not_called()
    update_card.assert_not_called()
    update_content.assert_any_call(update_content.call_args_list[0].args[0])
    content_request = update_content.call_args_list[0].args[0]
    assert content_request.card_id == "card_123"
    assert content_request.element_id == "content"
    assert content_request.request_body.content == "正在逐步输出"
    assert content_request.request_body.sequence == 1
    footer_request = update_content.call_args_list[1].args[0]
    assert footer_request.card_id == "card_123"
    assert footer_request.element_id == "footer"
    assert footer_request.request_body.sequence == 2
    assert "card_123" in adapter._streaming_card_state


@pytest.mark.asyncio
async def test_feishu_card_create_uses_raw_card_json_data_not_nested_card():
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    create = MagicMock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(card_id="card_raw_1")))
    adapter._client = SimpleNamespace(cardkit=SimpleNamespace(v1=SimpleNamespace(card=SimpleNamespace(create=create))))

    card_id = await adapter._create_card_message('{"schema":"2.0"}')

    assert card_id == "card_raw_1"
    body = create.call_args.args[0].request_body
    assert body.type == "card_json"
    assert body.data == '{"schema":"2.0"}'
    assert isinstance(body.data, str)


@pytest.mark.asyncio
async def test_feishu_edit_finalize_updates_cardkit_card_not_message_update(monkeypatch):
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._streaming_card_state = {"card_123": {"started_at": 100.0, "model": "provider/model-x"}}
    monkeypatch.setattr("gateway.platforms.feishu.time.monotonic", lambda: 103.24)
    adapter.format_message = lambda content: content
    adapter._build_card_update_request = lambda card_id, request_body: SimpleNamespace(
        card_id=card_id,
        request_body=request_body,
    )
    update_card = MagicMock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace()))
    settings_card = MagicMock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace()))
    update_message = MagicMock()
    adapter._client = SimpleNamespace(
        cardkit=SimpleNamespace(v1=SimpleNamespace(card=SimpleNamespace(update=update_card, settings=settings_card))),
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(update=update_message))),
    )

    result = await adapter.edit_message("chat_id", "card_123", "最终正文", finalize=True)

    assert result.success is True
    assert result.message_id == "card_123"
    update_message.assert_not_called()
    request = update_card.call_args.args[0]
    assert request.card_id == "card_123"
    assert request.request_body.sequence == 1
    card = __import__("json").loads(request.request_body.card.data)
    assert card["config"]["streaming_mode"] is False
    assert card["config"]["summary"]["content"] == "最终正文"
    footer_text = card["body"]["elements"][1]["content"]
    assert "Agent: main" in footer_text
    assert "Model: model-x" in footer_text
    assert "Elapsed: 3s" in footer_text
    assert "✅ 已完成" in footer_text
    assert "生成耗时实时更新" not in footer_text
    settings_card.assert_called_once()
    settings_request = settings_card.call_args.args[0]
    assert settings_request.card_id == "card_123"
    assert isinstance(settings_request.request_body.settings, str)
    settings_config = __import__("json").loads(settings_request.request_body.settings)
    assert settings_config["streaming_mode"] is False
    assert settings_config["summary"]["content"] == "最终正文"
    assert "card_123" not in adapter._streaming_card_state


@pytest.mark.asyncio
async def test_feishu_finalize_uses_short_summary_for_long_markdown(monkeypatch):
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._streaming_card_state = {"card_123": {"started_at": 100.0}}
    monkeypatch.setattr("gateway.platforms.feishu.time.monotonic", lambda: 103.24)
    adapter.format_message = lambda content: content
    adapter._build_card_update_request = lambda card_id, request_body: SimpleNamespace(
        card_id=card_id,
        request_body=request_body,
    )
    long_markdown = "# 标题\n" + "很长的正文 " * 80 + "\n```python\nprint('x')\n```"
    update_card = MagicMock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace()))
    settings_card = MagicMock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace()))
    adapter._client = SimpleNamespace(
        cardkit=SimpleNamespace(v1=SimpleNamespace(card=SimpleNamespace(update=update_card, settings=settings_card))),
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(update=MagicMock()))),
    )

    result = await adapter.edit_message("chat_id", "card_123", long_markdown, finalize=True)

    assert result.success is True
    card = __import__("json").loads(update_card.call_args.args[0].request_body.card.data)
    summary = card["config"]["summary"]["content"]
    assert len(summary) <= 200
    assert "\n" not in summary
    settings_config = __import__("json").loads(settings_card.call_args.args[0].request_body.settings)
    assert settings_config["summary"]["content"] == summary


@pytest.mark.asyncio
async def test_feishu_finalize_falls_back_to_element_updates_when_whole_card_update_fails(monkeypatch):
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._streaming_card_state = {"card_123": {"started_at": 100.0}}
    adapter._tool_progress_lines = {"chat_id": ["tool line"]}
    adapter._active_card_for_chat = {"chat_id": "card_123"}
    adapter._latest_card_content = {"card_123": "old"}
    adapter._tool_progress_last_update = {"chat_id": 1.0}
    monkeypatch.setattr("gateway.platforms.feishu.time.monotonic", lambda: 103.24)
    adapter.format_message = lambda content: content
    adapter._build_card_update_request = lambda card_id, request_body: SimpleNamespace(
        card_id=card_id,
        request_body=request_body,
    )
    update_card = MagicMock(return_value=SimpleNamespace(success=lambda: False, code=999, msg="invalid card"))
    update_content = MagicMock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace()))
    settings_card = MagicMock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace()))
    adapter._client = SimpleNamespace(
        cardkit=SimpleNamespace(
            v1=SimpleNamespace(
                card_element=SimpleNamespace(content=update_content),
                card=SimpleNamespace(update=update_card, settings=settings_card),
            )
        ),
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(update=MagicMock()))),
    )

    result = await adapter.edit_message("chat_id", "card_123", "最终正文", finalize=True)

    assert result.success is True
    assert update_card.call_count == 1
    assert update_content.call_count == 2
    assert update_content.call_args_list[0].args[0].element_id == "content"
    assert update_content.call_args_list[1].args[0].element_id == "footer"
    assert "✅ 已完成" in update_content.call_args_list[1].args[0].request_body.content
    settings_card.assert_called_once()
    assert "card_123" not in adapter._streaming_card_state
    assert "chat_id" not in adapter._active_card_for_chat


@pytest.mark.asyncio
async def test_feishu_append_mode_still_uses_element_updates_so_client_streams(monkeypatch):
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._append_mode = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._streaming_card_state = {
        "card_append_1": {
            "started_at": 100.0,
            "agent": "main",
            "model": "MiniMax-M2.7-highspeed",
            "provider": "minimax-cn",
        }
    }
    monkeypatch.setattr("gateway.platforms.feishu.time.monotonic", lambda: 108.0)
    adapter.format_message = lambda content: content
    adapter._build_card_update_request = lambda card_id, request_body: SimpleNamespace(
        card_id=card_id,
        request_body=request_body,
    )
    update_content = MagicMock()
    update_card = MagicMock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace()))
    update_message = MagicMock()
    adapter._client = SimpleNamespace(
        cardkit=SimpleNamespace(
            v1=SimpleNamespace(
                card_element=SimpleNamespace(content=update_content),
                card=SimpleNamespace(update=update_card),
            )
        ),
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(update=update_message))),
    )

    result = await adapter.edit_message("chat_id", "card_append_1", "1/2 旧进度\n2/2 新进度", finalize=False)

    assert result.success is True
    update_message.assert_not_called()
    update_card.assert_not_called()
    assert update_content.call_count == 2
    content_request = update_content.call_args_list[0].args[0]
    assert content_request.element_id == "content"
    assert content_request.request_body.content == "1/2 旧进度\n2/2 新进度"
    footer_request = update_content.call_args_list[1].args[0]
    assert footer_request.element_id == "footer"
    footer_text = footer_request.request_body.content
    assert "Agent: main" in footer_text
    assert "Model: MiniMax-M2.7-highspeed" in footer_text
    assert "Provider: minimax-cn" in footer_text
    assert "Elapsed: 8s" in footer_text
    assert footer_text.endswith("生成中")


def test_feishu_card_update_sequence_uses_small_incrementing_ints():
    adapter = FeishuAdapter.__new__(FeishuAdapter)

    first = adapter._build_card_update_body(data="{}")
    second = adapter._build_card_update_body(data="{}")

    assert first.sequence == 1
    assert second.sequence == 2


@pytest.mark.asyncio
async def test_trailing_footer_uses_same_feishu_card_send_path_when_stream_already_sent():
    footer = build_footer_line(
        user_config={
            "display": {
                "platforms": {
                    "feishu": {
                        "runtime_footer": {
                            "enabled": True,
                            "fields": ["model", "context_pct", "cwd"],
                        }
                    }
                }
            }
        },
        platform_key="feishu",
        model="provider/model-x",
        context_tokens=50,
        context_length=100,
        cwd="/tmp/project",
    )
    adapter = AsyncMock()
    adapter.send.return_value = SendResult(success=True, message_id="footer_msg")
    runner = FakeRunner(adapter)
    source = SimpleNamespace(platform=Platform.FEISHU, chat_id="chat_id")
    event = SimpleNamespace(source=source)
    agent_result = {"already_sent": True, "failed": False}

    await runner._deliver_media_from_response("正文", event, adapter)
    await adapter.send(source.chat_id, footer)

    adapter.send.assert_awaited_once_with("chat_id", footer)
    assert "model-x" in footer
    assert "50%" in footer


@pytest.mark.asyncio
async def test_stream_commentary_appends_semantic_progress_instead_of_sending_bubble():
    adapter = AsyncMock()
    adapter.MAX_MESSAGE_LENGTH = 8000
    adapter.SUPPORTS_MESSAGE_EDITING = True
    adapter.REQUIRES_EDIT_FINALIZE = True
    adapter.append_execution_progress.return_value = SendResult(success=True, message_id="card_progress")
    adapter.send.return_value = SendResult(success=True, message_id="separate_msg")

    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat_id",
        config=StreamConsumerConfig(edit_interval=999, buffer_threshold=999, cursor=""),
        metadata={"model": "provider/model-x"},
    )

    ok = await consumer._send_commentary("已完成苹果中文翻译，继续处理流式卡片")

    assert ok is True
    adapter.append_execution_progress.assert_awaited_once_with(
        "chat_id",
        "已完成苹果中文翻译，继续处理流式卡片",
        metadata={"model": "provider/model-x"},
    )
    adapter.send.assert_not_awaited()
    assert consumer.already_sent is False
    assert consumer._message_id == "card_progress"


@pytest.mark.asyncio
async def test_stream_commentary_preserves_card_for_later_final_edit():
    adapter = AsyncMock()
    adapter.MAX_MESSAGE_LENGTH = 8000
    adapter.SUPPORTS_MESSAGE_EDITING = True
    adapter.REQUIRES_EDIT_FINALIZE = True
    adapter.append_execution_progress.return_value = SendResult(success=True, message_id="card_progress")
    adapter.edit_message.return_value = SendResult(success=True, message_id="card_progress")
    adapter.send.return_value = SendResult(success=True, message_id="separate_msg")

    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat_id",
        config=StreamConsumerConfig(edit_interval=999, buffer_threshold=999, cursor=""),
        metadata={"model": "provider/model-x"},
    )

    consumer.on_commentary("第一段人类可读进度")
    consumer.on_commentary("第二段人类可读进度")
    consumer.on_delta("最终答复")
    consumer.finish()
    await consumer.run()

    assert adapter.append_execution_progress.await_count == 2
    adapter.send.assert_not_awaited()
    assert adapter.edit_message.await_count == 2
    final_edit = adapter.edit_message.await_args
    assert final_edit.kwargs == {
        "chat_id": "chat_id",
        "message_id": "card_progress",
        "content": "最终答复",
        "finalize": True,
    }
    assert all(call.kwargs["message_id"] == "card_progress" for call in adapter.edit_message.await_args_list)
    assert consumer.final_response_sent is True


@pytest.mark.asyncio
async def test_stream_commentary_survives_segment_break_and_finalizes_same_card():
    adapter = AsyncMock()
    adapter.MAX_MESSAGE_LENGTH = 8000
    adapter.SUPPORTS_MESSAGE_EDITING = True
    adapter.REQUIRES_EDIT_FINALIZE = True
    adapter.PRESERVE_STREAM_TARGET_ACROSS_SEGMENTS = True
    adapter.append_execution_progress.return_value = SendResult(success=True, message_id="card_progress")
    adapter.edit_message.return_value = SendResult(success=True, message_id="card_progress")
    adapter.send.return_value = SendResult(success=True, message_id="separate_msg")

    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat_id",
        config=StreamConsumerConfig(edit_interval=999, buffer_threshold=999, cursor=""),
        metadata={"model": "provider/model-x"},
    )

    consumer.on_commentary("先汇报计划，随后执行工具")
    consumer.on_delta(None)  # Tool/segment boundary must not orphan the CardKit progress card.
    consumer.on_delta("最终答复")
    consumer.finish()
    await consumer.run()

    adapter.append_execution_progress.assert_awaited_once_with(
        "chat_id",
        "先汇报计划，随后执行工具",
        metadata={"model": "provider/model-x"},
    )
    adapter.send.assert_not_awaited()
    assert adapter.edit_message.await_count == 2
    assert all(call.kwargs["message_id"] == "card_progress" for call in adapter.edit_message.await_args_list)
    assert adapter.edit_message.await_args_list[-1].kwargs == {
        "chat_id": "chat_id",
        "message_id": "card_progress",
        "content": "最终答复",
        "finalize": True,
    }
    assert consumer.final_response_sent is True


@pytest.mark.asyncio
async def test_stream_delta_segment_break_does_not_finalize_preserved_card_early():
    adapter = AsyncMock()
    adapter.MAX_MESSAGE_LENGTH = 8000
    adapter.SUPPORTS_MESSAGE_EDITING = True
    adapter.REQUIRES_EDIT_FINALIZE = True
    adapter.PRESERVE_STREAM_TARGET_ACROSS_SEGMENTS = True
    adapter.send.return_value = SendResult(success=True, message_id="card_stream")
    adapter.edit_message.return_value = SendResult(success=True, message_id="card_stream")

    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat_id",
        config=StreamConsumerConfig(edit_interval=999, buffer_threshold=999, cursor=""),
        metadata={"model": "provider/model-x"},
    )

    consumer.on_delta("前半部分")
    consumer.on_delta(None)  # Tool/segment boundary: keep card open.
    consumer.on_delta("最终答复")
    consumer.finish()
    await consumer.run()

    assert adapter.send.await_count == 1
    assert adapter.send.await_args.kwargs["metadata"] == {
        "model": "provider/model-x",
        "_hermes_stream_preview": True,
    }
    assert adapter.edit_message.await_count == 2
    first_edit = adapter.edit_message.await_args_list[0]
    final_edit = adapter.edit_message.await_args_list[-1]
    assert first_edit.kwargs["message_id"] == "card_stream"
    assert first_edit.kwargs["finalize"] is False
    assert final_edit.kwargs == {
        "chat_id": "chat_id",
        "message_id": "card_stream",
        "content": "最终答复",
        "finalize": True,
    }
    assert consumer.final_response_sent is True


@pytest.mark.asyncio
async def test_feishu_append_execution_progress_uses_same_card_without_tool_noise():
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._render_mode = "card"
    adapter._always_card = True
    adapter._card_footer_elapsed = True
    adapter._card_footer_status = True
    adapter._streaming_card_state = {}
    adapter._execution_progress_lines = {}
    adapter._tool_progress_lines = {}
    adapter._latest_card_content = {}
    adapter._active_card_for_chat = {}
    adapter._tool_progress_interval = 0
    adapter.format_message = lambda content: content
    adapter.truncate_message = lambda content, max_length: [content]
    adapter._client = SimpleNamespace(
        cardkit=SimpleNamespace(
            v1=SimpleNamespace(
                card=SimpleNamespace(update=MagicMock(return_value=SimpleNamespace(success=lambda: True))),
                card_element=SimpleNamespace(content=MagicMock(return_value=SimpleNamespace(success=lambda: True))),
            )
        ),
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(update=MagicMock()))),
    )
    adapter._feishu_send_with_retry = AsyncMock(
        return_value=SimpleNamespace(success=lambda: True, hermes_feishu_card_id="card_semantic_1")
    )

    first = await adapter.append_execution_progress(
        "chat_id",
        "已完成苹果中文翻译，继续处理流式卡片",
        metadata={"model": "provider/model-x", "provider": "test"},
    )

    assert first.success is True
    assert first.message_id == "card_semantic_1"
    payload = adapter._feishu_send_with_retry.await_args.kwargs["payload"]
    assert "已完成苹果中文翻译" in payload
    assert "terminal" not in payload
    assert "collapsible_panel" not in payload
    assert adapter._tool_progress_lines == {}
    assert adapter._active_card_for_chat["chat_id"] == "card_semantic_1"

    second = await adapter.append_execution_progress(
        "chat_id",
        "第二段人类可读进度",
        metadata={"model": "provider/model-x", "provider": "test"},
    )

    assert second.success is True
    update_request = adapter._client.cardkit.v1.card.update.call_args.args[0]
    updated_card = __import__("json").loads(update_request.request_body.card.data)
    content_element = next(
        element for element in updated_card["body"]["elements"] if element.get("element_id") == "content"
    )
    assert content_element["content"] == "已完成苹果中文翻译，继续处理流式卡片\n...\n第二段人类可读进度\n..."
    assert all(element.get("tag") != "collapsible_panel" for element in updated_card["body"]["elements"])
    assert adapter._tool_progress_lines == {}

    adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="card_semantic_1"))
    result = await adapter.send("chat_id", "最终答复开始", metadata={"model": "provider/model-x"})

    assert result.success is True
    assert result.message_id == "card_semantic_1"
    adapter.edit_message.assert_awaited_once_with(
        "chat_id",
        "card_semantic_1",
        "最终答复开始",
        finalize=True,
    )
