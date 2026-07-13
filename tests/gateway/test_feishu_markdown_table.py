"""Tests for rendering Feishu outbound Markdown tables as card tables."""

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from tests.gateway._plugin_adapter_loader import load_plugin_adapter


_feishu_mod = load_plugin_adapter("feishu")
FeishuAdapter = _feishu_mod.FeishuAdapter


class TestFeishuMarkdownTablePayload(unittest.TestCase):
    def _adapter(self):
        from gateway.config import PlatformConfig
        return FeishuAdapter(PlatformConfig())

    def _interactive_card(self, content: str):
        adapter = self._adapter()
        msg_type, payload = adapter._build_outbound_payload(content)
        self.assertEqual(msg_type, "interactive")
        return json.loads(payload)

    @patch.dict(os.environ, {}, clear=True)
    def test_build_outbound_payload_returns_interactive_card_for_simple_table(self):
        card = self._interactive_card(
            "Intro\n\n"
            "| 指标 | 状态 | 说明 |\n"
            "| --- | :---: | ---: |\n"
            "| CPU | 正常 | 12% |\n"
            "| Mem | 偏高 | 82% |"
        )

        self.assertTrue(card["config"]["wide_screen_mode"])
        self.assertTrue(card["config"]["update_multi"])
        self.assertEqual(card["config"]["locales"], ["zh_cn", "en_us"])
        table = card["elements"][1]
        self.assertEqual(table["tag"], "table")
        self.assertEqual(table["page_size"], 10)
        self.assertEqual(
            [column["display_name"] for column in table["columns"]],
            ["指标", "状态", "说明"],
        )
        self.assertEqual(table["rows"][0], {"col_0": "CPU", "col_1": "正常", "col_2": "12%"})
        self.assertEqual(table["rows"][1], {"col_0": "Mem", "col_1": "偏高", "col_2": "82%"})

    @patch.dict(os.environ, {}, clear=True)
    def test_prose_before_and_after_table_are_markdown_elements_in_order(self):
        card = self._interactive_card(
            "Before paragraph\n\n"
            "| A | B |\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n\n"
            "After paragraph"
        )

        self.assertEqual(
            [element["tag"] for element in card["elements"]],
            ["markdown", "table", "markdown"],
        )
        self.assertEqual(card["elements"][0]["content"], "Before paragraph")
        self.assertEqual(card["elements"][2]["content"], "After paragraph")

    @patch.dict(os.environ, {}, clear=True)
    def test_fenced_code_block_table_is_not_converted(self):
        adapter = self._adapter()
        content = "```markdown\n| A | B |\n| --- | --- |\n| 1 | 2 |\n```"

        msg_type, payload = adapter._build_outbound_payload(content)

        self.assertEqual(msg_type, "post")
        self.assertIn("| A | B |", payload)

    @patch.dict(os.environ, {}, clear=True)
    def test_escaped_pipe_and_inline_code_pipe_do_not_split_cells(self):
        card = self._interactive_card(
            "| Name | Expr |\n"
            "| --- | --- |\n"
            "| A\\|B | `x|y` |\n"
            "| short | one | extra |"
        )

        table = card["elements"][0]
        self.assertEqual(table["rows"][0], {"col_0": "A|B", "col_1": "`x|y`"})
        self.assertEqual(table["rows"][1], {"col_0": "short", "col_1": "one | extra"})

    @patch.dict(os.environ, {}, clear=True)
    def test_alignment_separator_maps_to_columns(self):
        card = self._interactive_card(
            "| L | C | R |\n"
            "| --- | :---: | ---: |\n"
            "| a | b | c |"
        )

        aligns = [
            column["horizontal_align"]
            for column in card["elements"][0]["columns"]
        ]
        self.assertEqual(aligns, ["left", "center", "right"])

    @patch.dict(os.environ, {}, clear=True)
    def test_heading_prose_is_rendered_as_sized_card_text(self):
        card = self._interactive_card(
            "# Main Title\n"
            "### Section Title\n\n"
            "| A | B |\n"
            "| --- | --- |\n"
            "| 1 | 2 |"
        )

        self.assertEqual([element["tag"] for element in card["elements"]], ["div", "div", "table"])
        self.assertEqual(card["elements"][0]["text"]["content"], "**Main Title**")
        self.assertEqual(card["elements"][0]["text"]["text_size"], "heading-1")
        self.assertEqual(card["elements"][1]["text"]["content"], "**Section Title**")
        self.assertEqual(card["elements"][1]["text"]["text_size"], "heading-3")

    @patch.dict(os.environ, {}, clear=True)
    def test_h5_heading_prose_gets_heading_size_in_card(self):
        card = self._interactive_card(
            "Intro\n\n"
            "##### 1. TODE80 / TODE90 的共同规则\n"
            "| 项 | 规则 |\n"
            "| --- | --- |\n"
            "| 容器名 | `teap_${new_branch}` |"
        )

        self.assertEqual([element["tag"] for element in card["elements"]], ["markdown", "div", "table"])
        self.assertEqual(card["elements"][1]["text"]["content"], "**1. TODE80 / TODE90 的共同规则**")
        self.assertEqual(card["elements"][1]["text"]["text_size"], "heading")

    @patch.dict(os.environ, {}, clear=True)
    def test_code_block_headings_are_not_normalized_in_card_markdown(self):
        card = self._interactive_card(
            "```markdown\n# Keep Literal\n```\n\n"
            "| A | B |\n"
            "| --- | --- |\n"
            "| 1 | 2 |"
        )

        self.assertIn("# Keep Literal", card["elements"][0]["content"])

    @patch.dict(os.environ, {}, clear=True)
    def test_content_without_table_preserves_existing_post_and_text_behavior(self):
        adapter = self._adapter()

        post_type, post_payload = adapter._build_outbound_payload("Hello **bold**")
        text_type, text_payload = adapter._build_outbound_payload("Hello plain")

        self.assertEqual(post_type, "post")
        self.assertIn("Hello **bold**", post_payload)
        self.assertEqual(text_type, "text")
        self.assertEqual(json.loads(text_payload), {"text": "Hello plain"})

    @patch.dict(os.environ, {}, clear=True)
    def test_allows_five_tables_in_one_card(self):
        content = "\n\n".join(
            f"Table {index}\n\n| A | B |\n| --- | --- |\n| {index} | ok |"
            for index in range(1, 6)
        )

        card = self._interactive_card(content)

        self.assertEqual(sum(1 for element in card["elements"] if element["tag"] == "table"), 5)

    @patch.dict(os.environ, {}, clear=True)
    def test_over_limit_table_count_falls_back_to_text(self):
        adapter = self._adapter()
        content = "\n\n".join(
            f"Table {index}\n\n| A | B |\n| --- | --- |\n| {index} | ok |"
            for index in range(1, 7)
        )

        msg_type, payload = adapter._build_outbound_payload(content)

        self.assertEqual(msg_type, "text")
        self.assertIn("Table 6", payload)

    @patch.dict(os.environ, {}, clear=True)
    def test_allows_fifteen_column_table(self):
        headers = [f"H{i}" for i in range(15)]
        content = (
            "| " + " | ".join(headers) + " |\n"
            + "| " + " | ".join(["---"] * 15) + " |\n"
            + "| " + " | ".join(["v"] * 15) + " |"
        )

        card = self._interactive_card(content)

        self.assertEqual(len(card["elements"][0]["columns"]), 15)

    @patch.dict(os.environ, {}, clear=True)
    def test_over_limit_table_falls_back_to_text_not_post(self):
        adapter = self._adapter()
        headers = [f"H{i}" for i in range(16)]
        content = (
            "**Intro**\n"
            + "| " + " | ".join(headers) + " |\n"
            + "| " + " | ".join(["---"] * 16) + " |\n"
            + "| " + " | ".join(["v"] * 16) + " |"
        )

        msg_type, payload = adapter._build_outbound_payload(content)
        edit_type, _ = adapter._build_outbound_payload(content, allow_interactive=False)

        self.assertEqual(msg_type, "text")
        self.assertEqual(edit_type, "text")
        self.assertIn("H15", payload)

    @patch.dict(os.environ, {}, clear=True)
    def test_tilde_fenced_code_block_table_is_not_converted(self):
        adapter = self._adapter()
        content = "~~~markdown\n| A | B |\n| --- | --- |\n| 1 | 2 |\n~~~"

        msg_type, payload = adapter._build_outbound_payload(content)

        self.assertEqual(msg_type, "text")
        self.assertIn("| A | B |", payload)

    @patch.dict(os.environ, {}, clear=True)
    def test_inline_code_or_escaped_pipe_before_rule_is_not_table(self):
        adapter = self._adapter()

        inline_type, _ = adapter._build_outbound_payload("Use `x|y`\n---")
        escaped_type, _ = adapter._build_outbound_payload("foo\\|bar\n---")

        self.assertEqual(inline_type, "post")
        self.assertEqual(escaped_type, "post")

    @patch.dict(os.environ, {}, clear=True)
    def test_backslashes_and_multibacktick_code_spans_are_preserved(self):
        card = self._interactive_card(
            "| Path | Expr |\n"
            "| --- | --- |\n"
            "| C:\\temp | ``x|y`` |"
        )

        table = card["elements"][0]
        self.assertEqual(table["rows"][0], {"col_0": "C:\\temp", "col_1": "``x|y``"})
    @patch.dict(os.environ, {}, clear=True)
    def test_text_with_inline_or_escaped_pipe_after_table_stays_prose(self):
        card = self._interactive_card(
            "| A | B |\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n"
            "Note `x|y` and A\\|B after table"
        )

        self.assertEqual([element["tag"] for element in card["elements"]], ["table", "markdown"])
        self.assertEqual(card["elements"][0]["rows"], [{"col_0": "1", "col_1": "2"}])
        self.assertEqual(card["elements"][1]["content"], "Note `x|y` and A\\|B after table")


class TestFeishuMarkdownTableSendEdit(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_interactive_send_failure_falls_back_to_plain_text(self):
        from gateway.config import PlatformConfig
        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = SimpleNamespace()
        response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="om_fallback"),
        )
        adapter._feishu_send_with_retry = AsyncMock(
            side_effect=[RuntimeError("interactive rejected"), response]
        )

        result = asyncio.run(
            adapter.send(
                chat_id="oc_chat",
                content="| A | B |\n| --- | --- |\n| 1 | 2 |",
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_fallback")
        self.assertEqual(
            [call.kwargs["msg_type"] for call in adapter._feishu_send_with_retry.call_args_list],
            ["interactive", "text"],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_interactive_response_failure_falls_back_to_text(self):
        from gateway.config import PlatformConfig
        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = SimpleNamespace()
        interactive_rejected = SimpleNamespace(success=lambda: False, msg="card rejected")
        text_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="om_text_fallback"),
        )
        adapter._feishu_send_with_retry = AsyncMock(
            side_effect=[
                interactive_rejected,
                text_response,
            ]
        )

        result = asyncio.run(
            adapter.send(
                chat_id="oc_chat",
                content="| A | B |\n| --- | --- |\n| 1 | 2 |",
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_text_fallback")
        self.assertEqual(
            [call.kwargs["msg_type"] for call in adapter._feishu_send_with_retry.call_args_list],
            ["interactive", "text"],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_over_limit_tables_fall_back_to_single_text_message(self):
        from gateway.config import PlatformConfig
        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = SimpleNamespace()
        response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_text"))
        adapter._feishu_send_with_retry = AsyncMock(return_value=response)
        content = "\n\n".join(
            f"Table {index}\n\n| A | B |\n| --- | --- |\n| {index} | ok |"
            for index in range(1, 7)
        )

        result = asyncio.run(adapter.send(chat_id="oc_chat", content=content, reply_to="om_user_message"))

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_text")
        self.assertEqual(
            [(call.kwargs["msg_type"], call.kwargs["reply_to"]) for call in adapter._feishu_send_with_retry.call_args_list],
            [("text", "om_user_message")],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_interactive_reply_rejection_falls_back_to_text_reply(self):
        from gateway.config import PlatformConfig

        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = SimpleNamespace()
        interactive_rejected = SimpleNamespace(success=lambda: False, code=230011, msg="reply target missing")
        text_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="om_text_reply"),
        )
        adapter._feishu_send_with_retry = AsyncMock(side_effect=[interactive_rejected, text_response])

        result = asyncio.run(
            adapter.send(
                chat_id="oc_chat",
                content="| A | B |\n| --- | --- |\n| 1 | 2 |",
                reply_to="om_user_message",
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_text_reply")
        self.assertEqual(
            [(call.kwargs["msg_type"], call.kwargs["reply_to"]) for call in adapter._feishu_send_with_retry.call_args_list],
            [("interactive", "om_user_message"), ("text", "om_user_message")],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_interactive_reply_exception_falls_back_to_text_reply(self):
        from gateway.config import PlatformConfig

        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = SimpleNamespace()
        text_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="om_text_after_exception"),
        )
        adapter._feishu_send_with_retry = AsyncMock(side_effect=[RuntimeError("reply rejected"), text_response])

        result = asyncio.run(
            adapter.send(
                chat_id="oc_chat",
                content="| A | B |\n| --- | --- |\n| 1 | 2 |",
                reply_to="om_user_message",
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_text_after_exception")
        self.assertEqual(
            [(call.kwargs["msg_type"], call.kwargs["reply_to"]) for call in adapter._feishu_send_with_retry.call_args_list],
            [("interactive", "om_user_message"), ("text", "om_user_message")],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_interactive_fallback_failure_returns_send_result(self):
        from gateway.config import PlatformConfig

        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = SimpleNamespace()
        adapter._feishu_send_with_retry = AsyncMock(
            side_effect=[RuntimeError("interactive rejected"), RuntimeError("text rejected")]
        )

        result = asyncio.run(
            adapter.send(
                chat_id="oc_chat",
                content="| A | B |\n| --- | --- |\n| 1 | 2 |",
                reply_to="om_user_message",
            )
        )

        self.assertFalse(result.success)
        self.assertIn("text rejected", result.error)
        self.assertEqual(
            [(call.kwargs["msg_type"], call.kwargs["reply_to"]) for call in adapter._feishu_send_with_retry.call_args_list],
            [("interactive", "om_user_message"), ("text", "om_user_message")],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_retry_does_not_turn_failed_interactive_reply_into_top_level_card(self):
        from gateway.config import PlatformConfig

        adapter = FeishuAdapter(PlatformConfig())
        rejected = SimpleNamespace(success=lambda: False, code=230011, msg="reply target missing")
        adapter._send_raw_message = AsyncMock(return_value=rejected)

        response = asyncio.run(
            adapter._feishu_send_with_retry(
                chat_id="oc_chat",
                msg_type="interactive",
                payload="{}",
                reply_to="om_user_message",
                metadata=None,
            )
        )

        self.assertIs(response, rejected)
        adapter._send_raw_message.assert_awaited_once()
        self.assertEqual(adapter._send_raw_message.call_args.kwargs["reply_to"], "om_user_message")

    @patch.dict(os.environ, {}, clear=True)
    def test_reply_in_thread_false_ignores_thread_metadata_fallback(self):
        from gateway.config import PlatformConfig

        adapter = FeishuAdapter(PlatformConfig(extra={"reply_in_thread": False}))
        message_api = SimpleNamespace(
            create=Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_top"))),
            reply=Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_reply"))),
        )
        adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=message_api)))

        response = asyncio.run(
            adapter._send_raw_message(
                chat_id="oc_chat",
                msg_type="text",
                payload=json.dumps({"text": "ok"}),
                reply_to=None,
                metadata={"thread_id": "omt_thread", "reply_to_message_id": "om_parent"},
            )
        )

        self.assertTrue(response.success())
        message_api.create.assert_called_once()
        message_api.reply.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_edit_message_does_not_attempt_interactive_update_for_table_content(self):
        from gateway.config import PlatformConfig

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _MessageAPI:
            def update(self, request):
                captured["request"] = request
                return SimpleNamespace(success=lambda: True)

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(message=_MessageAPI()))
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.object(adapter, "_run_blocking", side_effect=_direct):
            result = asyncio.run(
                adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_progress",
                    content="| A | B |\n| --- | --- |\n| 1 | 2 |",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["request"].request_body.msg_type, "text")
