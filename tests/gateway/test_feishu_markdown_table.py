"""Tests for rendering Feishu outbound Markdown tables as card tables."""

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


class TestFeishuMarkdownTablePayload(unittest.TestCase):
    def _adapter(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

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

        self.assertEqual(card["config"], {"wide_screen_mode": True})
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
    def test_content_without_table_preserves_existing_post_and_text_behavior(self):
        adapter = self._adapter()

        post_type, post_payload = adapter._build_outbound_payload("Hello **bold**")
        text_type, text_payload = adapter._build_outbound_payload("Hello plain")

        self.assertEqual(post_type, "post")
        self.assertIn("Hello **bold**", post_payload)
        self.assertEqual(text_type, "text")
        self.assertEqual(json.loads(text_payload), {"text": "Hello plain"})

    @patch.dict(os.environ, {}, clear=True)
    def test_over_limit_table_falls_back_to_text_not_post(self):
        adapter = self._adapter()
        headers = [f"H{i}" for i in range(13)]
        content = (
            "**Intro**\n"
            + "| " + " | ".join(headers) + " |\n"
            + "| " + " | ".join(["---"] * 13) + " |\n"
            + "| " + " | ".join(["v"] * 13) + " |"
        )

        msg_type, payload = adapter._build_outbound_payload(content)
        edit_type, _ = adapter._build_outbound_payload(content, allow_interactive=False)

        self.assertEqual(msg_type, "text")
        self.assertEqual(edit_type, "text")
        self.assertIn("H12", payload)

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
        from gateway.platforms.feishu import FeishuAdapter

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
        from gateway.platforms.feishu import FeishuAdapter

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
    def test_edit_message_does_not_attempt_interactive_update_for_table_content(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

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

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_progress",
                    content="| A | B |\n| --- | --- |\n| 1 | 2 |",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["request"].request_body.msg_type, "text")
