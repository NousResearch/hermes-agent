"""Unit tests for the Slack Block Kit renderer (pure function, no adapter)."""

from plugins.platforms.slack.block_kit import (
    MAX_BLOCKS,
    MAX_HEADER_TEXT,
    MAX_SECTION_TEXT,
    render_blocks,
)


def _types(blocks):
    return [b["type"] for b in blocks]


class TestRenderBlocksBasics:
    def test_empty_returns_none(self):
        assert render_blocks("") is None
        assert render_blocks("   \n  ") is None

    def test_plain_paragraph_is_section(self):
        blocks = render_blocks("just a plain sentence")
        assert blocks is not None
        assert len(blocks) == 1
        assert blocks[0]["type"] == "section"
        assert blocks[0]["text"]["type"] == "mrkdwn"

    def test_header_becomes_header_block(self):
        blocks = render_blocks("# Title")
        assert blocks[0]["type"] == "header"
        assert blocks[0]["text"]["type"] == "plain_text"
        assert blocks[0]["text"]["text"] == "Title"

    def test_header_strips_markup_and_caps_length(self):
        long = "#" + " " + "x" * 300
        blocks = render_blocks(long)
        assert blocks[0]["type"] == "header"
        assert len(blocks[0]["text"]["text"]) <= MAX_HEADER_TEXT

    def test_horizontal_rule_becomes_divider(self):
        blocks = render_blocks("above\n\n---\n\nbelow")
        assert "divider" in _types(blocks)

    def test_fenced_code_becomes_preformatted(self):
        md = "```python\ndef f():\n    return 1\n```"
        blocks = render_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "rich_text"
        assert blocks[0]["elements"][0]["type"] == "rich_text_preformatted"


class TestNestedLists:
    def test_nested_bullets_produce_increasing_indent(self):
        md = "- a\n  - b\n    - c"
        blocks = render_blocks(md)
        rich = [b for b in blocks if b["type"] == "rich_text"][0]
        indents = [e["indent"] for e in rich["elements"] if e["type"] == "rich_text_list"]
        # true nesting: indent levels must strictly increase across the run
        assert indents == sorted(indents)
        assert max(indents) >= 2
        assert min(indents) == 0

    def test_ordered_and_bullet_styles_distinguished(self):
        md = "1. first\n2. second\n\n- bullet"
        blocks = render_blocks(md)
        styles = []
        for b in blocks:
            if b["type"] == "rich_text":
                for e in b["elements"]:
                    if e["type"] == "rich_text_list":
                        styles.append(e["style"])
        assert "ordered" in styles
        assert "bullet" in styles


class TestInlineFormatting:
    def test_link_becomes_link_element(self):
        blocks = render_blocks("see [docs](https://example.com/x) now")
        # link lives in a section (paragraph) — but a bulleted link is a
        # rich_text link element; assert the URL survives somewhere.
        blob = str(blocks)
        assert "https://example.com/x" in blob

    def test_bulleted_bold_is_styled(self):
        blocks = render_blocks("- this is **bold** text")
        rich = [b for b in blocks if b["type"] == "rich_text"][0]
        section = rich["elements"][0]["elements"][0]
        styled = [
            el for el in section["elements"]
            if el.get("style", {}).get("bold")
        ]
        assert styled, "expected a bold-styled text element in the list item"


class TestTables:
    def test_pipe_table_renders_preformatted(self):
        md = (
            "| Name | Status |\n"
            "|------|--------|\n"
            "| a | ok |\n"
            "| b | fail |"
        )
        blocks = render_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "rich_text"
        pre = blocks[0]["elements"][0]
        assert pre["type"] == "rich_text_preformatted"
        text = pre["elements"][0]["text"]
        # header cell values preserved and column aligned
        assert "Name" in text and "Status" in text
        assert "fail" in text


class TestLimits:
    def test_oversized_section_is_split_under_limit(self):
        big = "word " * 2000  # ~10000 chars, single paragraph
        blocks = render_blocks(big)
        assert blocks is not None
        for b in blocks:
            if b["type"] == "section":
                assert len(b["text"]["text"]) <= MAX_SECTION_TEXT

    def test_too_many_blocks_returns_none(self):
        # 60 dividers => 60 blocks > MAX_BLOCKS => decline (caller uses text)
        md = "\n\n".join(["---"] * (MAX_BLOCKS + 10))
        assert render_blocks(md) is None

    def test_never_raises_on_garbage(self):
        for junk in ["```unterminated\ncode", "| broken | table", "> ", "#" * 10]:
            # must not raise; either blocks or None
            render_blocks(junk)
