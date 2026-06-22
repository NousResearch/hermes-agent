"""Tests for `FeishuAdapter._build_outbound_payload` routing.

Covers the fix for hermes-agent #26658 / #27529: messages containing markdown
tables should be sent as ``post`` + ``tag: md`` (which Feishu renders
correctly across all current clients), not force-downgraded to plain
``text``.
"""

import json
import unittest


class TestOutboundPayloadRouting(unittest.TestCase):
    def _make_adapter(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        return FeishuAdapter(PlatformConfig())

    def test_plain_text_routes_to_text(self):
        adapter = self._make_adapter()
        msg_type, payload = adapter._build_outbound_payload("Just a plain line.")
        self.assertEqual(msg_type, "text")
        self.assertEqual(json.loads(payload)["text"], "Just a plain line.")

    def test_markdown_without_table_routes_to_post(self):
        adapter = self._make_adapter()
        content = "Some **bold** text and a [link](https://example.com)."
        msg_type, _ = adapter._build_outbound_payload(content)
        self.assertEqual(msg_type, "post")

    def test_markdown_table_routes_to_post(self):
        """Regression test for #26658 / #27529.

        Tables must go to ``post`` (not ``text``) so Feishu renders them
        properly using the ``tag: md`` element. Forcing ``text`` was the
        old workaround for a Feishu rendering bug that has since been
        fixed upstream.
        """
        adapter = self._make_adapter()
        content = (
            "Look at this:\n\n"
            "| a | b |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
        )
        msg_type, payload = adapter._build_outbound_payload(content)
        self.assertEqual(msg_type, "post")
        # The post payload should carry the table text verbatim inside a
        # ``tag: md`` element so Feishu's renderer can format it.
        decoded = json.loads(payload)
        rows = decoded["zh_cn"]["content"]
        rendered = json.dumps(rows, ensure_ascii=False)
        self.assertIn("| a | b |", rendered)
        self.assertIn('"tag": "md"', rendered)

    def test_table_only_message_still_routes_to_post(self):
        """Table-only content (no surrounding prose) also goes to post."""
        adapter = self._make_adapter()
        content = (
            "| col1 | col2 |\n"
            "|------|------|\n"
            "| x    | y    |\n"
        )
        msg_type, _ = adapter._build_outbound_payload(content)
        self.assertEqual(msg_type, "post")

    def test_table_with_inline_markdown_routes_to_post(self):
        adapter = self._make_adapter()
        content = (
            "Summary below:\n\n"
            "| 维度 | 状态 |\n"
            "|------|------|\n"
            "| **粗体** | [链接](https://example.com) |\n\n"
            "End of summary."
        )
        msg_type, _ = adapter._build_outbound_payload(content)
        self.assertEqual(msg_type, "post")


if __name__ == "__main__":
    unittest.main()
