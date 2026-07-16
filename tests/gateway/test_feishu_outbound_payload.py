"""Tests for Feishu outbound payload routing — especially markdown table handling."""

import json
import sys
from pathlib import Path

import pytest

# Ensure the plugin adapter is importable
_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from plugins.platforms.feishu.adapter import (
    FeishuAdapter,
    _build_markdown_post_payload,
    _MARKDOWN_TABLE_RE,
)


# ---------------------------------------------------------------------------
# Helper: extract the payload kind + parsed body from _build_outbound_payload
# ---------------------------------------------------------------------------

def _route(content: str) -> tuple[str, dict]:
    """Return (msg_type, parsed_json_body) for the given content."""
    adapter = object.__new__(FeishuAdapter)
    msg_type, body_str = adapter._build_outbound_payload(content)
    return msg_type, json.loads(body_str)


# ---------------------------------------------------------------------------
# _MARKDOWN_TABLE_RE regex
# ---------------------------------------------------------------------------

class TestMarkdownTableRegex:
    def test_simple_table(self):
        t = "| a | b |\n|---|---|\n| 1 | 2 |"
        assert _MARKDOWN_TABLE_RE.search(t)

    def test_table_with_heading_row(self):
        t = "| Name | Score |\n| --- | --- |\n| Alice | 90 |"
        assert _MARKDOWN_TABLE_RE.search(t)

    def test_no_match_on_plain_text(self):
        assert not _MARKDOWN_TABLE_RE.search("hello world")

    def test_no_match_on_single_pipe_line(self):
        assert not _MARKDOWN_TABLE_RE.search("use | as separator")


# ---------------------------------------------------------------------------
# _build_outbound_payload routing
# ---------------------------------------------------------------------------

class TestOutboundPayloadRouting:
    def test_table_routes_to_post(self):
        """Tables must go through 'post' (md element), not plain text."""
        table = "| Name | Value |\n| --- | --- |\n| foo | bar |"
        msg_type, body = _route(table)
        assert msg_type == "post", f"expected post, got {msg_type}"
        # The post body should contain an md element with the table text
        md_text = body["zh_cn"]["content"][0][0]["text"]
        assert "| Name | Value |" in md_text

    def test_heading_routes_to_post(self):
        msg_type, _ = _route("# Hello")
        assert msg_type == "post"

    def test_plain_text_stays_text(self):
        msg_type, body = _route("just plain text")
        assert msg_type == "text"
        assert body["text"] == "just plain text"

    def test_mixed_table_and_heading(self):
        content = "# Title\n\n| A | B |\n|---|---|\n| 1 | 2 |"
        msg_type, body = _route(content)
        assert msg_type == "post"
        md_text = body["zh_cn"]["content"][0][0]["text"]
        assert "| A | B |" in md_text

    def test_table_with_code_block(self):
        content = "```\ncode\n```\n\n| X | Y |\n|---|---|\n| 1 | 2 |"
        msg_type, body = _route(content)
        assert msg_type == "post"
        # Should have multiple rows (code block split)
        assert len(body["zh_cn"]["content"]) >= 2

    def test_empty_content(self):
        msg_type, body = _route("")
        assert msg_type == "text"
        assert body["text"] == ""


# ---------------------------------------------------------------------------
# _build_markdown_post_payload
# ---------------------------------------------------------------------------

class TestBuildMarkdownPostPayload:
    def test_returns_valid_json(self):
        result = _build_markdown_post_payload("hello")
        parsed = json.loads(result)
        assert "zh_cn" in parsed
        assert "content" in parsed["zh_cn"]

    def test_table_in_md_element(self):
        table = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = _build_markdown_post_payload(table)
        parsed = json.loads(result)
        md_text = parsed["zh_cn"]["content"][0][0]["text"]
        assert "| A | B |" in md_text
