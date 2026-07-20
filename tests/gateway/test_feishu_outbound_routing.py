"""Regression tests for Feishu outbound payload routing."""

import json
import os
import unittest
from unittest.mock import patch


class TestFeishuOutboundPayloadRouting(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_plain_text_routes_to_text(self) -> None:
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        content = "Just a plain line."
        adapter = FeishuAdapter(PlatformConfig())

        msg_type, payload = adapter._build_outbound_payload(content)

        self.assertEqual(msg_type, "text")
        self.assertEqual(json.loads(payload), {"text": content})

    @patch.dict(os.environ, {}, clear=True)
    def test_table_only_content_routes_to_markdown_post(self) -> None:
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        content = "| col1 | col2 |\n|------|------|\n| x    | y    |"
        adapter = FeishuAdapter(PlatformConfig())

        msg_type, payload = adapter._build_outbound_payload(content)

        self.assertEqual(msg_type, "post")
        self.assertEqual(
            json.loads(payload),
            {
                "zh_cn": {
                    "content": [[{"tag": "md", "text": content}]],
                }
            },
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_mixed_markdown_and_table_routes_to_markdown_post(self) -> None:
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        content = (
            "Summary **below**:\n\n"
            "| dimension | status |\n"
            "|-----------|--------|\n"
            "| **build** | [passing](https://example.com) |\n\n"
            "End of summary."
        )
        adapter = FeishuAdapter(PlatformConfig())

        msg_type, payload = adapter._build_outbound_payload(content)

        self.assertEqual(msg_type, "post")
        self.assertEqual(
            json.loads(payload),
            {
                "zh_cn": {
                    "content": [[{"tag": "md", "text": content}]],
                }
            },
        )
