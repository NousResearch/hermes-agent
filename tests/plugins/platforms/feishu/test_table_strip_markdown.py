"""Tests for Feishu adapter markdown handling in _build_outbound_payload.

Verifies that:
- Markdown table content falls back to text type (Feishu post doesn't render tables)
- When forced to text, residual markdown markers (** / ` / ~~) are stripped
  instead of being emitted verbatim
- Non-table content keeps its existing post / text branch behaviour

See issue: markdown tables in Feishu text fallback leak literal ** / ` markers
"""

import json

import pytest

from plugins.platforms.feishu.adapter import FeishuAdapter


class _StubAdapter(FeishuAdapter):
    """Bypass BasePlatformAdapter.__init__ — we only test the pure helper."""

    def __init__(self):
        # skip super().__init__ — only call the bound method
        pass


@pytest.fixture
def adapter():
    return _StubAdapter()


def test_table_with_bold_strips_double_star(adapter):
    content = "| **指标** | **值** |\n| --- | --- |\n| users | 100 |"
    msg_type, payload = adapter._build_outbound_payload(content)
    assert msg_type == "text", "Table content must force text type"
    decoded = json.loads(payload)
    assert "**" not in decoded["text"], "** must be stripped in text fallback"
    assert decoded["text"].startswith("|")


def test_table_with_inline_code_strips_backticks(adapter):
    content = "| name | snippet |\n| --- | --- |\n| foo | `bar` |"
    msg_type, payload = adapter._build_outbound_payload(content)
    assert msg_type == "text"
    decoded = json.loads(payload)
    assert "`" not in decoded["text"], "backticks must be stripped"


def test_table_with_strikethrough_strips_tildes(adapter):
    content = "| status |\n| --- |\n| ~~done~~ |"
    msg_type, payload = adapter._build_outbound_payload(content)
    decoded = json.loads(payload)
    assert "~~" not in decoded["text"], "~~ must be stripped"
    assert "done" in decoded["text"]


def test_table_rows_preserved_after_strip(adapter):
    """Behavior contract: stripping markdown must not destroy table structure."""
    content = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    _, payload = adapter._build_outbound_payload(content)
    decoded = json.loads(payload)
    # row count preserved (no row collapsing)
    assert decoded["text"].count("\n") == 2
    # column count preserved per row (each row has 2 separators)
    for line in decoded["text"].split("\n"):
        assert line.count("|") >= 2


def test_plain_text_no_markdown_stays_text(adapter):
    """No markdown hint → text type, content unchanged."""
    content = "正常文本没有 markdown"
    msg_type, payload = adapter._build_outbound_payload(content)
    assert msg_type == "text"
    assert json.loads(payload)["text"] == content


def test_post_branch_unaffected_by_table_fix(adapter):
    """Non-table content with markdown hint still goes through post branch."""
    content = "这是 **加粗** 测试"
    msg_type, payload = adapter._build_outbound_payload(content)
    assert msg_type == "post", "non-table markdown must keep post branch"
    # post branch payload uses zh_cn.content shape
    decoded = json.loads(payload)
    assert "zh_cn" in decoded