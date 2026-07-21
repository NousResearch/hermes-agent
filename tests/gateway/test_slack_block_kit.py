"""Unit tests for the Slack Block Kit renderer (pure function, no adapter)."""

from plugins.platforms.slack.block_kit import (
    MAX_BLOCKS,
    MAX_HEADER_TEXT,
    MAX_SECTION_TEXT,
    _display_width,
    _render_table,
    blocks_fallback_text,
    demote_tables,
    has_table_block,
    render_blocks,
    segment_blocks,
)


def _types(blocks):
    return [b["type"] for b in blocks]


class TestRenderBlocksBasics:
    def test_empty_returns_none(self):
        assert render_blocks("") is None
        assert render_blocks("   \n  ") is None

    def test_plain_paragraph_is_rich_text(self):
        blocks = render_blocks("just a plain sentence")
        assert blocks is not None
        assert len(blocks) == 1
        assert blocks[0]["type"] == "rich_text"
        section = blocks[0]["elements"][0]
        assert section["type"] == "rich_text_section"
        assert section["elements"][0] == {"type": "text", "text": "just a plain sentence"}

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

    def test_cjk_adjacent_bold_gets_explicit_style(self):
        """Regression: mrkdwn refuses ``*bold*`` glued to CJK punctuation
        (no word boundary), so ``**7.39 GB**，`` showed literal asterisks.
        rich_text style objects don't depend on boundary parsing.
        """
        blocks = render_blocks("预计扫描 **7.39 GB**，仍在 10 GB 限制内")
        section = blocks[0]["elements"][0]
        bold = [
            el for el in section["elements"]
            if el.get("style", {}).get("bold")
        ]
        assert bold and bold[0]["text"] == "7.39 GB"
        assert "**" not in str(blocks)

    def test_user_mention_becomes_user_element(self):
        blocks = render_blocks("ping <@U012AB3CD> about this")
        section = blocks[0]["elements"][0]
        users = [el for el in section["elements"] if el["type"] == "user"]
        assert users == [{"type": "user", "user_id": "U012AB3CD"}]

    def test_broadcast_and_channel_entities(self):
        blocks = render_blocks("<!here> see <#C024BE7LR|general>")
        section = blocks[0]["elements"][0]
        types = {el["type"] for el in section["elements"]}
        assert "broadcast" in types
        assert {"type": "channel", "channel_id": "C024BE7LR"} in section["elements"]

    def test_bare_url_becomes_link_element(self):
        blocks = render_blocks("详见 https://example.com/x。")
        section = blocks[0]["elements"][0]
        links = [el for el in section["elements"] if el["type"] == "link"]
        # trailing CJK full stop is prose, not part of the URL
        assert links == [{"type": "link", "url": "https://example.com/x"}]
        tail = section["elements"][-1]
        assert tail == {"type": "text", "text": "。"}

    def test_manual_slack_link_with_label(self):
        blocks = render_blocks("open <https://e.io/a|the dashboard> now")
        section = blocks[0]["elements"][0]
        links = [el for el in section["elements"] if el["type"] == "link"]
        assert links == [{"type": "link", "url": "https://e.io/a", "text": "the dashboard"}]

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
        # Slack requires each column setting to be an object; an explicit
        # left alignment is valid while null entries are rejected by the API.
        assert cs[0] == {"align": "left"}
        assert cs[1] == {"align": "center"}
        assert cs[2] == {"align": "right"}

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

    def test_display_width_counts_cjk_as_two_columns(self):
        assert _display_width("LCP") == 3
        assert _display_width("指标") == 4
        assert _display_width("p75 值") == 6

    def test_display_width_ignores_zero_width_combiners(self):
        # decomposed é (e + combining acute) is one printed column, not two;
        # a ZWJ between two letters adds nothing.
        assert _display_width("é") == 1
        assert _display_width("café") == 4  # precomposed é stays one column
        assert _display_width("a‍b") == 2  # ZWJ is zero-width

    def test_cjk_fallback_alignment_uses_display_width(self):
        rows = [
            "| 指标 | 判断 |",
            "| LCP | 良好 |",
            "| INP | Needs Improvement |",
        ]
        out = _render_table(rows)
        lines = out.split("\n")
        data_lines = [lines[0]] + lines[2:]  # lines[1] is the header underline
        # The first column separator must sit at the same display column on
        # every row; with len()-based padding the CJK header row drifts.
        prefix_cols = {
            _display_width(line[: line.index("|")]) for line in data_lines
        }
        assert len(prefix_cols) == 1
        # Header underline spans the display width of the widest cell (指标 = 4).
        assert lines[1].startswith("-" * 4)

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
    def test_oversized_paragraph_is_split_under_limit(self):
        big = "word " * 2000  # ~10000 chars, single paragraph
        blocks = render_blocks(big)
        assert blocks is not None
        assert len(blocks) > 1  # split across multiple paragraph blocks
        for b in blocks:
            assert b["type"] == "rich_text"
            total = sum(
                len(el.get("text", ""))
                for sec in b["elements"]
                for el in sec["elements"]
            )
            assert total <= MAX_SECTION_TEXT

    def test_over_max_blocks_returns_full_list_for_segmentation(self):
        # 60 dividers => 60 blocks. The renderer no longer declines; callers
        # segment the list into multiple messages via segment_blocks().
        md = "\n\n".join(["---"] * (MAX_BLOCKS + 10))
        blocks = render_blocks(md)
        assert blocks is not None
        assert len(blocks) == MAX_BLOCKS + 10

    def test_never_raises_on_garbage(self):
        for junk in ["```unterminated\ncode", "| broken | table", "> ", "#" * 10]:
            # must not raise; either blocks or None
            render_blocks(junk)


def _section(i):
    return {"type": "section", "text": {"type": "mrkdwn", "text": f"s{i}"}}


class TestSegmentBlocks:
    def test_under_cap_single_segment(self):
        blocks = [_section(i) for i in range(10)]
        assert segment_blocks(blocks) == [blocks]

    def test_hard_cut_at_cap_without_boundary(self):
        blocks = [_section(i) for i in range(120)]
        segments = segment_blocks(blocks)
        assert all(len(s) <= MAX_BLOCKS for s in segments)
        # nothing lost, order preserved
        flat = [b for s in segments for b in s]
        assert flat == blocks

    def test_prefers_header_boundary(self):
        header = {"type": "header", "text": {"type": "plain_text", "text": "h"}}
        blocks = [_section(i) for i in range(40)] + [header] + [
            _section(i) for i in range(19)
        ]  # 60 blocks, header at index 40 (inside the trailing-third window)
        segments = segment_blocks(blocks)
        assert len(segments[0]) == 40
        assert segments[1][0]["type"] == "header"

    def test_leading_divider_dropped_after_cut(self):
        divider = {"type": "divider"}
        blocks = [_section(i) for i in range(50)] + [divider] + [_section(99)]
        segments = segment_blocks(blocks)
        # cut lands on the divider; the next segment must not open with it —
        # the message break itself is the separator
        assert len(segments) == 2
        assert segments[1] == [_section(99)]

    def test_fallback_text_extracts_content(self):
        md = "# 标题\n\n预计扫描 **7.39 GB**，详见 https://e.io/x"
        segments = segment_blocks(render_blocks(md))
        text = blocks_fallback_text(segments[0])
        assert "标题" in text
        assert "7.39 GB" in text
        assert "**" not in text  # plain projection, no markup

    def test_fallback_text_never_empty(self):
        assert blocks_fallback_text([{"type": "divider"}]).strip() != "" or (
            blocks_fallback_text([{"type": "divider"}]) == " "
        )


class TestProgressContextBlocks:
    def test_empty_returns_none(self):
        from plugins.platforms.slack.block_kit import progress_context_blocks

        assert progress_context_blocks("") is None
        assert progress_context_blocks("   \n ") is None

    def test_fence_collapses_to_inline_code(self):
        from plugins.platforms.slack.block_kit import progress_context_blocks

        content = "💻 terminal\n```\nset -e && make build\nmake test\n```"
        [block] = progress_context_blocks(content)
        assert block["type"] == "context"
        text = block["elements"][0]["text"]
        # multi-line command collapses to its first line as inline code;
        # & is mrkdwn-escaped (Slack renders &amp; back as & inside code)
        assert "`set -e &amp;&amp; make build …`" in text
        assert "```" not in text

    def test_keeps_only_trailing_lines_with_elision_note(self):
        from plugins.platforms.slack.block_kit import progress_context_blocks

        content = "\n".join(f"📖 Reading file{i}.csv" for i in range(25))
        [block] = progress_context_blocks(content, max_lines=10)
        text = block["elements"][0]["text"]
        lines = text.split("\n")
        assert len(lines) == 11  # elision note + 10 kept lines
        assert "15 earlier steps" in lines[0]
        assert lines[-1].endswith("file24.csv")

    def test_zero_keeps_all_lines(self):
        from plugins.platforms.slack.block_kit import progress_context_blocks

        content = "\n".join(f"line{i}" for i in range(25))
        [block] = progress_context_blocks(content, max_lines=0)
        assert len(block["elements"][0]["text"].split("\n")) == 25

    def test_control_chars_escaped(self):
        from plugins.platforms.slack.block_kit import progress_context_blocks

        [block] = progress_context_blocks("fetch <https://e.io> & parse")
        text = block["elements"][0]["text"]
        assert "&lt;" in text and "&amp;" in text


class TestAmbiguousWidthAlignment:
    def test_ambiguous_width_wide_in_cjk_table(self):
        # Arrows/✓/… are East Asian "Ambiguous": two columns in a CJK font.
        # A table that contains CJK must count them as two or rows drift.
        rows = ["| 名称 | 状态 |", "| 首页→详情 | ✓ 好 |", "| API… | 差 |"]
        out = _render_table(rows)
        lines = out.split("\n")
        data_lines = [lines[0]] + lines[2:]
        cols = {
            _display_width(line[: line.index("|")], wide_ambiguous=True)
            for line in data_lines
        }
        assert len(cols) == 1  # first separator aligned across all rows

    def test_ascii_table_keeps_ambiguous_one_column(self):
        assert _display_width("a→b") == 3  # no CJK context: → is one column
        assert _display_width("甲→乙", wide_ambiguous=True) == 6  # CJK: → is two


class TestLongParagraphSplit:
    def test_split_preserves_bold_across_seam(self):
        # A >3000-char paragraph full of bold spans must split WITHOUT
        # bisecting a ``**...**`` span into literal asterisks.
        big = "说明 **关键指标** 数据 " * 300
        blocks = render_blocks(big)
        assert len(blocks) > 1
        assert "**" not in str(blocks)  # no leaked markup at any seam
        bolds = {
            el["text"]
            for b in blocks
            for sec in b["elements"]
            for el in sec["elements"]
            if el.get("style", {}).get("bold")
        }
        assert bolds == {"关键指标"}  # every bold span intact


class TestDemoteTables:
    _TABLE_MD = "| 指标 | 判断 |\n|------|------|\n| LCP | 良好 |\n| INP | 需改进 |"

    def test_has_table_block(self):
        blocks = render_blocks(self._TABLE_MD)
        assert has_table_block(blocks)
        assert not has_table_block(render_blocks("just text"))

    def test_demote_converts_table_to_monospace(self):
        blocks = render_blocks(self._TABLE_MD)
        demoted = demote_tables(blocks)
        assert not has_table_block(demoted)
        # the table became an aligned monospace preformatted block
        pre = [b for b in demoted if b["type"] == "rich_text"]
        assert pre and pre[0]["elements"][0]["type"] == "rich_text_preformatted"
        text = pre[0]["elements"][0]["elements"][0]["text"]
        assert "指标" in text and "LCP" in text and "|" in text  # a grid, not raw md

    def test_demote_keeps_other_blocks(self):
        blocks = render_blocks(f"# Title\n\n{self._TABLE_MD}\n\nafter")
        demoted = demote_tables(blocks)
        types = [b["type"] for b in demoted]
        assert "header" in types  # non-table blocks untouched
        assert not has_table_block(demoted)


class TestNoEmptyTextElements:
    """Slack rejects a zero-length rich_text ``text`` element ('must be more
    than 0 characters') and one bad element fails the WHOLE blocks payload —
    the message then degrades to plain text. Empty table cells / code fences
    must never emit an empty text element."""

    @staticmethod
    def _has_empty_text(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "text" and obj.get("text", "") == "":
                return True
            return any(TestNoEmptyTextElements._has_empty_text(v) for v in obj.values())
        if isinstance(obj, list):
            return any(TestNoEmptyTextElements._has_empty_text(x) for x in obj)
        return False

    def test_empty_table_cells_no_empty_text(self):
        md = "| A | B |\n|---|---|\n| x |  |\n|  | y |"
        assert not self._has_empty_text(render_blocks(md))

    def test_all_empty_row_no_empty_text(self):
        md = "| A | B |\n|---|---|\n|  |  |"
        assert not self._has_empty_text(render_blocks(md))

    def test_empty_code_fence_no_empty_text(self):
        assert not self._has_empty_text(render_blocks("```\n\n```"))

    def test_inline_elements_empty_string(self):
        from plugins.platforms.slack.block_kit import _inline_elements
        els = _inline_elements("")
        assert els and all(e.get("text") for e in els if e.get("type") == "text")
