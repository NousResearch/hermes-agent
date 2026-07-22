"""Unit tests for the Slack Block Kit renderer (pure function, no adapter)."""

from plugins.platforms.slack.block_kit import (
    MAX_BLOCKS,
    MAX_HEADER_TEXT,
    MAX_SECTION_TEXT,
    render_blocks,
    sanitize_blocks,
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

    def test_blank_line_separated_ordered_items_stay_in_one_list(self):
        """Regression: blank lines between ordered items must not reset numbering.

        Slack numbers each rich_text_list independently.  If blank lines break
        the list run, N items produce N separate lists each starting at 1.
        See: https://github.com/NousResearch/hermes-agent/issues/57076
        """
        md = "1. alpha\n\n1. beta\n\n1. gamma"
        blocks = render_blocks(md)
        rich = [b for b in blocks if b["type"] == "rich_text"][0]
        lists = [e for e in rich["elements"] if e["type"] == "rich_text_list"]
        # Must be ONE list with 3 items, not 3 separate single-item lists
        assert len(lists) == 1
        items = lists[0]["elements"]
        assert len(items) == 3

    def test_blank_separated_mixed_list_matches_contiguous_layout(self):
        """A blank line between different list kinds must render like the
        contiguous form: one rich_text block whose sub-lists split only on
        (indent, ordered) changes — not a separate block per item.
        """
        rich = [b for b in render_blocks("1. a\n\n- b") if b["type"] == "rich_text"]
        # Single rich_text block (matches contiguous "1. a\n- b"), two sub-lists
        assert len(rich) == 1
        styles = [e["style"] for e in rich[0]["elements"] if e["type"] == "rich_text_list"]
        assert styles == ["ordered", "bullet"]

    def test_blank_line_before_paragraph_ends_the_list(self):
        """A blank line followed by non-list content must still end the run,
        so a list → paragraph → list sequence stays three separate blocks.
        """
        blocks = render_blocks("1. a\n\nsome paragraph text\n\n1. b")
        lists = [
            e
            for b in blocks
            for e in b.get("elements", [])
            if e.get("type") == "rich_text_list"
        ]
        # Two independent single-item lists, not one merged three-item list
        assert [len(e["elements"]) for e in lists] == [1, 1]


class TestTables:
    def test_pipe_table_renders_native_table_block(self):
        md = (
            "| Name | Status |\n"
            "|------|--------|\n"
            "| a | ok |\n"
            "| b | fail |"
        )
        blocks = render_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "table"
        rows = blocks[0]["rows"]
        # header + 2 body rows, 2 columns each
        assert len(rows) == 3
        assert all(len(r) == 2 for r in rows)
        # cells are rich_text carrying the values
        assert str(rows[0]).count("Name") == 1
        assert "fail" in str(rows[2])

    def test_alignment_parsed_into_column_settings(self):
        md = (
            "| L | C | R |\n"
            "|:---|:--:|---:|\n"
            "| 1 | 2 | 3 |"
        )
        blocks = render_blocks(md)
        cs = blocks[0]["column_settings"]
        # Every provided entry must be a valid Slack column-settings object.
        # Left placeholders are explicit only when needed to preserve position.
        assert cs[0] == {"align": "left"}
        assert cs[1] == {"align": "center"}
        assert cs[2] == {"align": "right"}

    def test_default_trailing_column_settings_are_omitted(self):
        md = (
            "| L | R | L2 |\n"
            "|---|---:|---|\n"
            "| 1 | 2 | 3 |"
        )
        blocks = render_blocks(md)
        assert blocks is not None
        cs = blocks[0]["column_settings"]
        assert cs == [{"align": "left"}, {"align": "right"}]
        assert all(isinstance(item, dict) for item in cs)

    def test_all_default_table_omits_column_settings(self):
        md = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |"
        )
        blocks = render_blocks(md)
        assert blocks is not None
        assert "column_settings" not in blocks[0]

    def test_inline_formatting_inside_cells(self):
        md = (
            "| Item | Link |\n"
            "|------|------|\n"
            "| **bold** | [x](https://e.io) |"
        )
        blocks = render_blocks(md)
        body = blocks[0]["rows"][1]
        # bold styled text element in first cell
        bold = [
            el for el in body[0]["elements"][0]["elements"]
            if el.get("style", {}).get("bold")
        ]
        assert bold
        # link element in second cell
        links = [el for el in body[1]["elements"][0]["elements"] if el["type"] == "link"]
        assert links and links[0]["url"] == "https://e.io"

    def test_oversized_table_falls_back_to_monospace(self):
        # 120 rows > MAX_TABLE_ROWS -> monospace rich_text fallback, not a table
        big = "| a | b |\n|---|---|\n" + "\n".join(f"| x{i} | y |" for i in range(120))
        blocks = render_blocks(big)
        assert blocks[0]["type"] == "rich_text"  # preformatted fallback
        assert blocks[0]["elements"][0]["type"] == "rich_text_preformatted"

    def test_too_many_columns_falls_back_to_monospace(self):
        header = "|" + "|".join(f"c{i}" for i in range(25)) + "|"
        sep = "|" + "|".join("-" for _ in range(25)) + "|"
        row = "|" + "|".join("v" for _ in range(25)) + "|"
        blocks = render_blocks(f"{header}\n{sep}\n{row}")
        assert blocks[0]["type"] == "rich_text"

    def test_escaped_pipe_not_a_column_separator(self):
        md = (
            "| Expr | Meaning |\n"
            "|------|--------|\n"
            "| a \\| b | or |"
        )
        blocks = render_blocks(md)
        assert blocks[0]["type"] == "table"
        # the escaped-pipe cell stays a single cell containing a literal pipe
        body = blocks[0]["rows"][1]
        assert len(body) == 2
        assert "|" in str(body[0])


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


class TestEmptyContentGuards:
    """Empty content must never produce a Slack-rejected (invalid_blocks) payload.

    Slack rejects a rich_text_section / rich_text_preformatted /
    rich_text_quote whose ``elements`` is empty or contains a zero-length
    ``text`` element, and a ``header`` whose plain_text is empty. Each guard
    below corresponds to a real chat.postMessage rejection observed in
    production ("missing element" / "must be more than 0 characters").
    """

    @staticmethod
    def _assert_schema_valid(blocks):
        def walk(o):
            if isinstance(o, dict):
                if o.get("type") in (
                    "rich_text_section", "rich_text_preformatted", "rich_text_quote"
                ):
                    assert o.get("elements"), f"empty {o['type']} elements"
                if o.get("type") == "text":
                    assert len(o.get("text", "")) > 0, "zero-length text element"
                if o.get("type") == "header":
                    assert o["text"]["text"], "empty plain_text header"
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(blocks)

    def test_ragged_and_empty_table_cells_are_schema_valid(self):
        # Blank middle cell + ragged short row (padded with "") must not emit
        # an empty section or a 0-char text element.
        md = (
            "| x | y | z |\n"
            "| --- | --- | --- |\n"
            "| 1 |  | 3 |\n"   # blank middle cell
            "| 4 |"           # ragged row -> padded with empty cells
        )
        blocks = render_blocks(md)
        assert blocks[0]["type"] == "table"
        self._assert_schema_valid(blocks)

    def test_empty_code_fence_quote_and_list_item_are_schema_valid(self):
        # Empty fenced code block (common around empty tool output), blank
        # quote line, and empty list item must all stay schema-valid.
        md = "```\n```\n\n> \n\n- \n- real item"
        blocks = render_blocks(md)
        assert blocks is not None
        self._assert_schema_valid(blocks)

    def test_multiline_quote_preserves_newline_separators(self):
        # _quote_block separates lines with length-1 "\n" text elements; the
        # guard must KEEP them so a multi-line blockquote stays multi-line.
        blocks = render_blocks("> alpha\n> bravo")
        quote = None
        for b in blocks:
            for el in b.get("elements", []):
                if isinstance(el, dict) and el.get("type") == "rich_text_quote":
                    quote = el
        assert quote is not None, "no rich_text_quote produced"
        texts = [e.get("text") for e in quote["elements"] if e.get("type") == "text"]
        assert "\n" in texts, "newline separator dropped from multi-line quote"
        assert any("alpha" in (t or "") for t in texts)
        assert any("bravo" in (t or "") for t in texts)

    def test_emphasis_only_header_is_dropped_not_empty(self):
        # "# ***" reduces to "" after marker-strip; an empty plain_text header
        # is rejected by Slack, so the header is skipped entirely.
        blocks = render_blocks("# ***\n\nreal body")
        assert not any(b.get("type") == "header" for b in blocks)
        self._assert_schema_valid(blocks)

    def test_normal_content_unaffected(self):
        # Guard must not alter well-formed content.
        md = "# Title\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n> quoted\n\n- item"
        blocks = render_blocks(md)
        assert any(b.get("type") == "header" for b in blocks)
        assert any(b.get("type") == "table" for b in blocks)
        self._assert_schema_valid(blocks)


class TestSanitizeBlocks:
    """Outbound boundary clamp: one bad block must never fail the whole call.

    Regression coverage for the invalid_blocks / msg_too_long bug class
    (#56615 null column_settings, #62054 / #53693 >3000-char sections on
    approval chat.update after HTML-escaping inflation).
    """

    def test_none_and_empty_return_none(self):
        assert sanitize_blocks(None) is None
        assert sanitize_blocks([]) is None

    def test_valid_payload_passes_through(self):
        blocks = render_blocks("# Title\n\nbody text\n\n- item")
        assert sanitize_blocks(blocks) == blocks

    def test_oversized_section_text_is_clamped(self):
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": "x" * 3500}},
        ]
        out = sanitize_blocks(blocks)
        assert len(out[0]["text"]["text"]) <= MAX_SECTION_TEXT
        assert out[0]["text"]["text"].endswith("…")

    def test_html_escape_inflated_approval_update_is_clamped(self):
        # #53693 / #62054: send path budgeted the RAW text to <=3000, but the
        # interaction payload echoes it back HTML-escaped (& -> &amp;) so the
        # chat.update section exceeds the cap.
        inflated = "a" * 2990 + "&amp;" * 10  # 3040 chars
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": inflated}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": "✅ ok"}]},
        ]
        out = sanitize_blocks(blocks)
        assert len(out[0]["text"]["text"]) <= MAX_SECTION_TEXT
        # context block untouched
        assert out[1] == blocks[1]

    def test_null_column_settings_entries_are_fixed(self):
        # #56615: Slack rejects null entries in table column_settings.
        table = {
            "type": "table",
            "rows": [[{"type": "rich_text", "elements": []}]],
            "column_settings": [None, {"align": "center"}, None],
        }
        out = sanitize_blocks([table])
        cs = out[0]["column_settings"]
        assert cs == [{}, {"align": "center"}]
        assert all(isinstance(c, dict) for c in cs)

    def test_all_null_column_settings_are_dropped(self):
        table = {
            "type": "table",
            "rows": [[{"type": "rich_text", "elements": []}]],
            "column_settings": [None, None],
        }
        out = sanitize_blocks([table])
        assert "column_settings" not in out[0]

    def test_empty_blocks_are_dropped(self):
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": "   "}},
            {"type": "rich_text", "elements": []},
            {"type": "actions", "elements": []},
            {"type": "context", "elements": []},
            {"type": "header", "text": {"type": "plain_text", "text": ""}},
            {"type": "table", "rows": []},
            {"type": "section", "text": {"type": "mrkdwn", "text": "keep me"}},
        ]
        out = sanitize_blocks(blocks)
        assert out == [blocks[-1]]

    def test_all_invalid_returns_none_for_plain_text_fallback(self):
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": ""}}]
        assert sanitize_blocks(blocks) is None

    def test_oversized_header_is_clamped(self):
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": "h" * 200}},
        ]
        out = sanitize_blocks(blocks)
        assert len(out[0]["text"]["text"]) <= MAX_HEADER_TEXT

    def test_payload_capped_at_50_blocks(self):
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"b{i}"}}
            for i in range(60)
        ]
        out = sanitize_blocks(blocks)
        assert len(out) == MAX_BLOCKS

    def test_never_raises_on_garbage(self):
        assert sanitize_blocks([{"no_type": True}, "not-a-dict", 42]) is None
