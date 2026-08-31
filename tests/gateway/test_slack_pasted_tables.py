"""Slack pasted-table recovery (ported from qwibitai/nanoclaw#3666).

Slack represents a pasted table as ``table`` blocks — usually nested inside
``attachments[].blocks[]``, sometimes top-level. The table appears in neither
the message ``text`` nor the file list, so before this port the agent
received the sentence before the table and nothing else.
"""

from plugins.platforms.slack.adapter import (
    _SLACK_TABLE_MAX_CHARS,
    _collect_slack_table_cell_text,
    _extract_additional_text_from_slack_blocks,
    _extract_text_from_slack_attachments,
    _extract_text_from_slack_blocks,
    _render_slack_table_block,
    _serialize_slack_blocks_for_agent,
)


def _raw_cell(text: str) -> dict:
    return {"type": "raw_text", "text": text}


def _rich_cell(text: str, bold: bool = False) -> dict:
    style = {"bold": True} if bold else {}
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_section",
                "elements": [{"type": "text", "text": text, "style": style}],
            }
        ],
    }


def _table_block(rows) -> dict:
    return {"type": "table", "rows": rows}


class TestCellText:
    def test_raw_text_cell(self):
        assert _collect_slack_table_cell_text(_raw_cell("Name")) == "Name"

    def test_rich_text_cell_collects_leaves(self):
        assert _collect_slack_table_cell_text(_rich_cell("Bold header", bold=True)) == (
            "Bold header"
        )

    def test_non_dict_cell_is_empty(self):
        assert _collect_slack_table_cell_text("stray") == ""
        assert _collect_slack_table_cell_text(None) == ""

    def test_list_of_nodes(self):
        cells = [_raw_cell("a"), _raw_cell("b")]
        assert _collect_slack_table_cell_text(cells) == "a b"


class TestRenderTableBlock:
    def test_projects_rows_pipe_separated(self):
        block = _table_block(
            [
                [_raw_cell("Name"), _raw_cell("Status")],
                [_raw_cell("Hermes"), _rich_cell("ok")],
            ]
        )
        assert _render_slack_table_block(block) == "Name | Status\nHermes | ok"

    def test_no_rows_returns_empty(self):
        assert _render_slack_table_block({"type": "table"}) == ""
        assert _render_slack_table_block({"type": "table", "rows": "bad"}) == ""
        assert _render_slack_table_block(_table_block([])) == ""

    def test_malformed_row_skipped(self):
        block = _table_block(["not-a-row", [_raw_cell("x"), _raw_cell("y")]])
        assert _render_slack_table_block(block) == "x | y"

    def test_empty_rows_dropped(self):
        block = _table_block([[_raw_cell(""), _raw_cell("")], [_raw_cell("k")]])
        assert _render_slack_table_block(block) == "k"

    def test_truncation_cap(self):
        big = _table_block([[_raw_cell("x" * 5000)] for _ in range(10)])
        out = _render_slack_table_block(big)
        assert out.endswith("[table truncated]")
        assert len(out) <= _SLACK_TABLE_MAX_CHARS


class TestBlockExtraction:
    def test_top_level_table_block_rendered(self):
        blocks = [_table_block([[_raw_cell("a"), _raw_cell("b")]])]
        assert _extract_text_from_slack_blocks(blocks) == "a | b"

    def test_table_alongside_rich_text(self):
        blocks = [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [{"type": "text", "text": "See table:"}],
                    }
                ],
            },
            _table_block([[_raw_cell("k"), _raw_cell("v")]]),
        ]
        out = _extract_text_from_slack_blocks(blocks)
        assert "See table:" in out
        assert "k | v" in out

    def test_additional_text_path_surfaces_table(self):
        # The live inbound path routes top-level blocks through
        # _extract_additional_text_from_slack_blocks with the flat text as
        # the dedupe reference — the table must survive that dedupe.
        blocks = [_table_block([[_raw_cell("col1"), _raw_cell("col2")]])]
        out = _extract_additional_text_from_slack_blocks(blocks, "intro sentence")
        assert "col1 | col2" in out


class TestAttachmentNestedTable:
    def test_attachment_blocks_table_recovered(self):
        # The real-world shape: pasted table arrives as
        # attachments[].blocks[] with type "table" and nothing in text.
        attachments = [
            {"blocks": [_table_block([[_raw_cell("Item"), _raw_cell("Qty")]])]}
        ]
        out = _extract_text_from_slack_attachments(attachments)
        assert "Item | Qty" in out


class TestSerializerSkipsTables:
    def test_table_block_not_json_dumped(self):
        # Table blocks are rendered as text; the JSON serializer must not
        # emit an empty {"type": "table"} husk for them.
        blocks = [_table_block([[_raw_cell("a")]])]
        assert _serialize_slack_blocks_for_agent(blocks) == ""

    def test_other_blocks_still_serialized(self):
        blocks = [
            _table_block([[_raw_cell("a")]]),
            {"type": "section", "text": {"type": "mrkdwn", "text": "hello"}},
        ]
        out = _serialize_slack_blocks_for_agent(blocks)
        assert "section" in out
        assert '"table"' not in out
