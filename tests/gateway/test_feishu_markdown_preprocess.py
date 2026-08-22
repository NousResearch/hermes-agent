"""Tests for `plugins.platforms.feishu.feishu_markdown_preprocess`.

Covers the 12 pitfalls called out in skill `feishu-markdown-preprocess` plus
the 11 representative content types from `references/render-validation-
fixtures.md`.  Each pitfall gets at least one targeted test; together they
exercise the full surface of `is_complex_markdown` and
`preprocess_to_post_payload`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from plugins.platforms.feishu import feishu_markdown_preprocess as pp  # noqa: E402


# --- Test 1: is_complex_markdown --------------------------------------------

class TestIsComplexMarkdown:
    def test_plain_text_returns_false(self):
        """Path A handles plain prose — no native path needed."""
        assert pp.is_complex_markdown("你好，世界。") is False

    def test_inline_emphasis_alone_returns_false(self):
        """Bold / italic / strike / code alone don't justify native."""
        assert pp.is_complex_markdown("这是 **重要通知** 和 *补充*。") is False
        assert pp.is_complex_markdown("运行命令：`pip install foo`") is False
        assert pp.is_complex_markdown("~~旧方案~~ 新方案") is False

    def test_fenced_code_triggers(self):
        assert pp.is_complex_markdown("```python\nx = 1\n```") is True

    def test_blockquote_triggers(self):
        assert pp.is_complex_markdown("> 必须本周完成") is True

    def test_horizontal_rule_triggers(self):
        assert pp.is_complex_markdown("above\n\n---\n\nbelow") is True

    def test_image_triggers(self):
        assert pp.is_complex_markdown("Look: ![alt](https://x)") is True

    def test_atx_heading_triggers(self):
        assert pp.is_complex_markdown("# Top\n\nbody") is True
        assert pp.is_complex_markdown("body\n\n## Subsection") is True

    def test_empty_returns_false(self):
        assert pp.is_complex_markdown("") is False

    def test_tables_do_not_trigger(self):
        """Path A handles GFM tables since #52786."""
        assert pp.is_complex_markdown("| col1 | col2 |\n|---|---|\n| a | b |") is False


# --- Test 2: over-escape pattern (Pitfall #1) -------------------------------

class TestOverEscapePattern:
    def test_compiles_in_python_312(self):
        """The pattern must use re.escape() — direct char-class construction
        fails on Python 3.12+."""
        assert pp._OVER_ESCAPED_RE is not None

    def test_strips_backslash_before_native_chars(self):
        text = r"\*literal asterisks\*"
        result = pp._OVER_ESCAPED_RE.sub(lambda m: m.group(0)[1:], text)
        assert result == "*literal asterisks*"


# --- Test 3: inline emphasis rendering (Pitfall #2, token pairs) -----------

class TestInlineEmphasis:
    def test_bold_emits_style_array(self):
        row = pp._render_inline("这是 **重要通知**")
        # Find the bold element
        bold = [e for e in row if e.get("style") == ["bold"]]
        assert len(bold) == 1
        assert bold[0]["text"] == "重要通知"

    def test_italic_emits_style_array(self):
        row = pp._render_inline("这是 *补充说明*")
        italic = [e for e in row if e.get("style") == ["italic"]]
        assert len(italic) == 1
        assert italic[0]["text"] == "补充说明"

    def test_strikethrough_emits_style_array(self):
        row = pp._render_inline("~~旧方案~~")
        strike = [e for e in row if e.get("style") == ["strikethrough"]]
        assert len(strike) == 1
        assert strike[0]["text"] == "旧方案"

    def test_inline_code_kept_intact(self):
        """Pitfall #2: inline code is opaque; backticks preserved."""
        row = pp._render_inline("运行命令：`pip install foo`")
        assert any(e.get("text") == "`pip install foo`" for e in row)

    def test_html_strings_do_not_appear_in_output(self):
        """Path B must never emit `<b>...</b>` HTML strings."""
        row = pp._render_inline("这是 **重要通知**")
        for e in row:
            assert "<b>" not in e.get("text", "")
            assert "<i>" not in e.get("text", "")
            assert "<s>" not in e.get("text", "")
            assert "<code>" not in e.get("text", "")

    def test_no_standalone_br_element(self):
        """Pitfall #11: soft/hard breaks must not become `tag: br`."""
        payload = pp.preprocess_to_post_payload("line one\nline two")
        assert payload is not None
        rows = payload["zh_cn"]["content"]
        flat = [e for row in rows for e in row]
        assert all(e.get("tag") != "br" for e in flat)


# --- Test 4: list rendering (Pitfall #3, level-based indent) ---------------

class TestListRendering:
    def test_bullet_list_indent(self):
        rows = pp._content_to_rows(
            "今天的 TODO:\n- 升级侧车\n- 测试双层架构\n- 检查飞书渲染"
        )
        # Expect: 1 paragraph + 3 bullet rows
        assert len(rows) >= 3
        bullet_rows = [r for r in rows if any("• " in e.get("text", "") for e in r)]
        assert len(bullet_rows) == 3

    def test_ordered_list_uses_numeric_prefix(self):
        rows = pp._content_to_rows("1. one\n2. two\n3. three")
        ordered_rows = [
            r for r in rows
            if any(re.match(r"\d+\.\s", e.get("text", "")) for e in r)
        ]
        assert len(ordered_rows) == 3
        prefixes = [
            next(e["text"] for e in r if re.match(r"\d+\.\s", e.get("text", "")))
            for r in ordered_rows
        ]
        assert prefixes[0].startswith("1.")
        assert prefixes[1].startswith("2.")
        assert prefixes[2].startswith("3.")

    def test_nested_list_indent(self):
        rows = pp._content_to_rows(
            "云南行程:\n- D3 大理\n  - 上午: 磻溪 S 湾\n  - 下午: 才村\n- D4 喜洲"
        )
        # Outer items should appear; nested items should have leading spaces
        flat = "\n".join(
            "".join(e.get("text", "") for e in r) for r in rows
        )
        assert "D3 大理" in flat
        assert "D4 喜洲" in flat
        # Nested bullets carry indent
        nested = [
            r for r in rows
            if any(("  • " in e.get("text", "") or "    " in e.get("text", ""))
                   for e in r)
        ]
        assert len(nested) >= 2, f"expected nested bullets, got {nested}"


# --- Test 5: blockquote (Pitfall #5, re-detection) --------------------------

class TestBlockquote:
    def test_standalone_blockquote_renders_with_bar(self):
        rows = pp._content_to_rows("> 必须本周完成\n> 不要拖到下周")
        flat = "".join(e.get("text", "") for r in rows for e in r)
        assert "│ 必须本周完成" in flat

    def test_merged_blockquote_redetected_in_paragraph(self):
        """Pitfall #5: markdown-it merges adjacent > lines into paragraph;
        our walker must re-detect."""
        rows = pp._content_to_rows("> 必须本周完成\n> 不要拖到下周")
        flat = "\n".join(
            "".join(e.get("text", "") for e in r) for r in rows
        )
        # Should be rendered as blockquote (with bar), not as prose with
        # raw "> " characters.
        assert "│ " in flat


# --- Test 6: hr handling (Pitfall #6) --------------------------------------

class TestHorizontalRule:
    def test_standalone_hr_renders_dashes(self):
        rows = pp._content_to_rows("above\n\n---\n\nbelow")
        hr_rows = [r for r in rows if any("──" in e.get("text", "") for e in r)]
        assert len(hr_rows) == 1

    def test_merged_hr_redetected(self):
        """Pitfall #6: `---` inside a paragraph re-detected as hr."""
        rows = pp._content_to_rows("前文\n---\n后文")
        flat = "\n".join(
            "".join(e.get("text", "") for e in r) for r in rows
        )
        assert "──" in flat


# --- Test 7: empty-element filtering (Pitfall #7) --------------------------

class TestEmptyElementFilter:
    def test_empty_text_elements_filtered(self):
        """Pitfall #7: in_blockquote branch must not produce empty elements."""
        row = pp._render_inline("")
        # All returned elements must have non-empty text OR be the
        # explicit empty sentinel {"text": ""} (one element max).
        assert all(e.get("text") != "" for e in row[1:])


# --- Test 8: heading rendering ----------------------------------------------

class TestHeadings:
    def test_heading_size_scales_with_level(self):
        for level in range(1, 7):
            hashes = "#" * level
            rows = pp._content_to_rows(f"{hashes} Title text")
            # First row should have size attribute matching the level
            first_row = rows[0]
            first_elem = first_row[0]
            expected_size = {1: 22, 2: 20, 3: 18, 4: 17, 5: 16, 6: 15}[level]
            assert first_elem.get("size") == expected_size, (
                f"level {level}: expected size {expected_size}, got {first_elem.get('size')}"
            )
            assert first_elem.get("style") == ["bold"]


# --- Test 9: code block rendering ------------------------------------------

class TestCodeBlock:
    def test_fenced_code_emits_code_block_element(self):
        payload = pp.preprocess_to_post_payload(
            "```python\ndef hello():\n    return 'hi'\n```"
        )
        assert payload is not None
        rows = payload["zh_cn"]["content"]
        code_rows = [
            r for r in rows
            if any(e.get("tag") == "code_block" for e in r)
        ]
        assert len(code_rows) == 1
        code_elem = next(e for e in code_rows[0] if e.get("tag") == "code_block")
        assert code_elem.get("language") == "PYTHON"
        assert "def hello" in code_elem.get("text", "")


# --- Test 10: complex combo (skill fixture #9) ------------------------------

class TestComplexCombo:
    def test_heading_list_code_blockquote_combo(self):
        content = (
            "# 云南行程概览\n"
            "\n"
            "## 关键时间\n"
            "\n"
            "- 7-26 09:30 高铁出发\n"
            "- 7-29 13:00 高铁返程\n"
            "\n"
            "## 注意事项\n"
            "\n"
            "```\n"
            "海拔 ≤2630m\n"
            "每日步行 ≤2.5km\n"
            "```\n"
            "\n"
            "> 老人和小孩优先级最高"
        )
        payload = pp.preprocess_to_post_payload(content)
        assert payload is not None
        rows = payload["zh_cn"]["content"]
        # Expect at least: heading, subheading, 2 bullets, code block, blockquote
        assert len(rows) >= 6
        flat = "\n".join(
            "".join(e.get("text", "") for e in r) for r in rows
        )
        assert "云南行程概览" in flat
        assert "高铁出发" in flat
        assert "海拔" in flat
        assert "│ 老人和小孩" in flat


# --- Test 11: public API error path -----------------------------------------

class TestErrorPath:
    def test_preprocess_returns_dict_for_complex(self):
        payload = pp.preprocess_to_post_payload("# Title\n\nbody")
        assert payload is not None
        assert "zh_cn" in payload
        assert "content" in payload["zh_cn"]

    def test_preprocess_handles_empty(self):
        assert pp.preprocess_to_post_payload("") is None

    def test_payload_is_json_serializable(self):
        payload = pp.preprocess_to_post_payload(
            "# Title\n\n- item one\n- item two\n\n```\ncode\n```"
        )
        assert payload is not None
        # Must round-trip through json.dumps for Feishu's API
        encoded = json.dumps(payload, ensure_ascii=False)
        decoded = json.loads(encoded)
        assert decoded == payload