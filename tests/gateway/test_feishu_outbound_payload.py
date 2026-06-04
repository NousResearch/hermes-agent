"""Tests for Feishu _build_outbound_payload markdown routing.

Verifies that markdown content (including tables) is always sent via
Feishu's post format with 'md' elements for proper rendering.
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter():
    """Create a minimal FeishuAdapter skeleton for testing _build_outbound_payload."""
    from gateway.platforms.feishu import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    return adapter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildOutboundPayload:
    """Verify _build_outbound_payload routes content to the correct msg_type."""

    def test_plain_text_returns_text(self):
        adapter = _make_adapter()
        msg_type, payload = adapter._build_outbound_payload("hello world")
        assert msg_type == "text"
        data = json.loads(payload)
        assert data["text"] == "hello world"

    def test_markdown_heading_returns_post(self):
        adapter = _make_adapter()
        msg_type, payload = adapter._build_outbound_payload("# Title\nSome content")
        assert msg_type == "post"
        data = json.loads(payload)
        rows = data["zh_cn"]["content"]
        assert len(rows) >= 1
        assert rows[0][0]["tag"] == "md"

    def test_markdown_table_with_heading_returns_post(self):
        """Table mixed with markdown hints (heading) must use post format."""
        adapter = _make_adapter()
        content = (
            "# Report\n\n"
            "| Name | Value |\n"
            "|------|-------|\n"
            "| foo  | 42    |\n"
            "| bar  | 99    |"
        )
        msg_type, payload = adapter._build_outbound_payload(content)
        assert msg_type == "post"
        data = json.loads(payload)
        rows = data["zh_cn"]["content"]
        assert rows[0][0]["tag"] == "md"
        assert "| Name | Value |" in rows[0][0]["text"]

    def test_pure_table_without_hints_returns_text(self):
        """A bare table with no other markdown syntax has no hints to trigger
        post format — same as before the fix.  Feishu renders bare pipe text
        acceptably in text mode."""
        adapter = _make_adapter()
        table_content = (
            "| Name | Value |\n"
            "|------|-------|\n"
            "| foo  | 42    |"
        )
        msg_type, payload = adapter._build_outbound_payload(table_content)
        assert msg_type == "text"

    def test_code_block_returns_post(self):
        adapter = _make_adapter()
        code_content = "Here is code:\n```python\nprint('hi')\n```\nDone."
        msg_type, payload = adapter._build_outbound_payload(code_content)
        assert msg_type == "post"
        data = json.loads(payload)
        rows = data["zh_cn"]["content"]
        assert len(rows) >= 2

    def test_bold_text_returns_post(self):
        adapter = _make_adapter()
        msg_type, payload = adapter._build_outbound_payload("This is **bold** text")
        assert msg_type == "post"

    def test_link_returns_post(self):
        adapter = _make_adapter()
        msg_type, payload = adapter._build_outbound_payload("Check [this](https://example.com)")
        assert msg_type == "post"

    def test_empty_content_returns_text(self):
        adapter = _make_adapter()
        msg_type, payload = adapter._build_outbound_payload("")
        assert msg_type == "text"

    def test_table_with_code_block_returns_post(self):
        """Table + code block: both should be in post format."""
        adapter = _make_adapter()
        content = (
            "| Col | Val |\n"
            "|-----|-----|\n"
            "| a   | 1   |\n\n"
            "```python\nprint('test')\n```"
        )
        msg_type, payload = adapter._build_outbound_payload(content)
        assert msg_type == "post"
        data = json.loads(payload)
        rows = data["zh_cn"]["content"]
        assert len(rows) >= 2

    def test_no_markdown_table_regex_leakage(self):
        """_MARKDOWN_TABLE_RE was removed — verify it no longer exists."""
        import gateway.platforms.feishu as feishu_mod
        assert not hasattr(feishu_mod, "_MARKDOWN_TABLE_RE")
