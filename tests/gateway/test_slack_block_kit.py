"""Unit tests for the Slack Block Kit renderer (pure function, no adapter)."""

from plugins.platforms.slack.block_kit import (
    MAX_BLOCKS,
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


    def test_header_becomes_header_block(self):
        blocks = render_blocks("# Title")
        assert blocks[0]["type"] == "header"
        assert blocks[0]["text"]["type"] == "plain_text"
        assert blocks[0]["text"]["text"] == "Title"


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


class TestInlineFormatting:

    def test_slack_mrkdwn_link_in_bullet_becomes_link_element(self):
        """rich_text lists must parse Slack <url|label>, not emit it as text.

        rich_blocks turns markdown bullets into rich_text_list. rich_text does
        not interpret mrkdwn, so <url|text> has to become type=link.
        """
        blocks = render_blocks(
            "- <https://example.com/x|GitLab #1> — allow `in_progress`"
        )
        assert blocks is not None
        rich = [b for b in blocks if b["type"] == "rich_text"][0]
        els = rich["elements"][0]["elements"][0]["elements"]
        links = [e for e in els if e.get("type") == "link"]
        assert len(links) == 1
        assert links[0]["url"] == "https://example.com/x"
        assert links[0]["text"] == "GitLab #1"
        assert not any("<https://" in (e.get("text") or "") for e in els)
        assert any(e.get("style", {}).get("code") for e in els)

    def test_slack_mentions_in_bullet_are_not_links(self):
        """Mentions must become mention elements, never link elements.

        rich_text does not interpret mrkdwn, so a token left as a text element
        renders literally in Slack (see #133315).
        """
        blocks = render_blocks("- ping <@U123> in <#C456>")
        assert blocks is not None
        els = blocks[0]["elements"][0]["elements"][0]["elements"]
        assert all(e.get("type") != "link" for e in els)
        assert {"type": "user", "user_id": "U123"} in els
        assert {"type": "channel", "channel_id": "C456"} in els
        assert not any("<@" in (e.get("text") or "") or "<#" in (e.get("text") or "") for e in els)


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


class TestSlackMentions:
    """Mention tokens must become rich_text user/channel/broadcast elements.

    Regression coverage for #133315: mentions inside rich_text (lists, quotes,
    table cells) were emitted as text elements, so Slack showed the raw token
    literally instead of a mention.
    """

    @staticmethod
    def _item_elements(blocks):
        rich = [b for b in blocks if b["type"] == "rich_text"][0]
        return rich["elements"][0]["elements"][0]["elements"]

    def test_channel_mention_with_label_keeps_only_the_id(self):
        blocks = render_blocks("- see <#C0C6N2RF5HS|general>")
        assert blocks is not None
        els = self._item_elements(blocks)
        assert {"type": "channel", "channel_id": "C0C6N2RF5HS"} in els
        assert not any("general" in (e.get("text") or "") for e in els)

    def test_broadcast_mentions_map_to_range(self):
        blocks = render_blocks("- <!here> <!channel> <!everyone>")
        assert blocks is not None
        els = self._item_elements(blocks)
        assert [e["range"] for e in els if e.get("type") == "broadcast"] == [
            "here",
            "channel",
            "everyone",
        ]

    def test_mention_in_quote_becomes_mention_element(self):
        blocks = render_blocks("> ping <@U123> now")
        assert blocks is not None
        quote = blocks[0]["elements"][0]
        assert quote["type"] == "rich_text_quote"
        assert {"type": "user", "user_id": "U123"} in quote["elements"]

    def test_mention_in_table_cell_becomes_mention_element(self):
        blocks = render_blocks("| who | note |\n| --- | --- |\n| <@U123> | ok |")
        assert blocks is not None
        assert blocks[0]["type"] == "table"
        cell = blocks[0]["rows"][1][0]
        assert {"type": "user", "user_id": "U123"} in cell["elements"][0]["elements"]

    def test_unmatched_angle_token_is_preserved_as_text(self):
        # Not a mention and not an autolink: must survive verbatim, not vanish.
        blocks = render_blocks("- a <b> c")
        assert blocks is not None
        els = self._item_elements(blocks)
        assert "a <b> c" in "".join(e.get("text") or "" for e in els)


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


    def test_oversized_table_falls_back_to_monospace(self):
        # 120 rows > MAX_TABLE_ROWS -> monospace rich_text fallback, not a table
        big = "| a | b |\n|---|---|\n" + "\n".join(f"| x{i} | y |" for i in range(120))
        blocks = render_blocks(big)
        assert blocks[0]["type"] == "rich_text"  # preformatted fallback
        assert blocks[0]["elements"][0]["type"] == "rich_text_preformatted"


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

    def test_too_many_blocks_returns_none(self):
        # 60 dividers => 60 blocks > MAX_BLOCKS => decline (caller uses text)
        md = "\n\n".join(["---"] * (MAX_BLOCKS + 10))
        assert render_blocks(md) is None


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


class TestSanitizeBlocks:
    """Outbound boundary clamp: one bad block must never fail the whole call.

    Regression coverage for the invalid_blocks / msg_too_long bug class
    (#56615 null column_settings, #62054 / #53693 >3000-char sections on
    approval chat.update after HTML-escaping inflation).
    """


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


class TestSplitTextFenceBalanced:
    """_split_text closes/reopens ``` fences at section chunk boundaries."""

    def test_fenced_split_every_chunk_balanced(self):
        from plugins.platforms.slack.block_kit import _split_text

        text = "```\n" + "\n".join("y" * 20 for _ in range(30)) + "\n```"
        chunks = _split_text(text, 100)
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks):
            assert chunk.count("```") % 2 == 0, (
                f"chunk {i} has unbalanced fences: {chunk[:60]!r}"
            )


