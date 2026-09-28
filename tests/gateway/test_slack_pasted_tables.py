"""Slack pasted-table recovery.

Slack represents a pasted table as ``table`` blocks — usually nested inside
``attachments[].blocks[]``, sometimes top-level. The table appears in neither the message
``text`` nor the file list, so the agent used to receive the sentence before it and nothing else.
"""

from plugins.platforms.slack.adapter import (
    _SLACK_TABLE_MAX_CHARS,
    SlackAdapter,
    _extract_additional_text_from_slack_blocks,
    _extract_text_from_slack_attachments,
    _render_slack_table_block,
    _serialize_slack_blocks_for_agent,
)


def _raw(text: str) -> dict:
    return {"type": "raw_text", "text": text}


def _rich(text: str) -> dict:
    return {"type": "rich_text", "elements": [
        {"type": "rich_text_section", "elements": [{"type": "text", "text": text, "style": {"bold": True}}]}]}


def _table(rows) -> dict:
    return {"type": "table", "rows": rows}


def test_render_handles_raw_rich_ragged_and_malformed_cells():
    block = _table([
        [_raw("Name"), _rich("Status")],
        "not-a-row",
        [_raw(""), None],
        [_raw("Hermes"), _rich("ok"), {"type": "mystery"}],
    ])
    assert _render_slack_table_block(block) == "Name | Status\nHermes | ok | "
    assert _render_slack_table_block({"type": "table"}) == ""
    assert _render_slack_table_block({"type": "table", "rows": "bad"}) == ""


def test_render_caps_huge_tables_with_visible_marker():
    out = _render_slack_table_block(_table([[_raw("x" * 5000)] for _ in range(10)]))
    assert out.endswith("[table truncated]")
    assert len(out) <= _SLACK_TABLE_MAX_CHARS


def test_top_level_table_survives_inbound_dedupe():
    # The live inbound path routes top-level blocks through the dedupe-against-flat-text helper.
    out = _extract_additional_text_from_slack_blocks(
        [_table([[_raw("col1"), _raw("col2")]])], "intro sentence")
    assert "col1 | col2" in out


def test_attachment_nested_table_reaches_live_and_history_paths():
    attachments = [{"blocks": [_table([[_raw("Item"), _raw("Qty")]])]}]
    assert "Item | Qty" in SlackAdapter._append_link_unfurls("intro", attachments)
    assert "Item | Qty" in _extract_text_from_slack_attachments(attachments)


def test_serializer_skips_table_husk_but_keeps_other_blocks():
    blocks = [_table([[_raw("a")]]), {"type": "section", "text": {"type": "mrkdwn", "text": "hello"}}]
    assert _serialize_slack_blocks_for_agent([blocks[0]]) == ""
    out = _serialize_slack_blocks_for_agent(blocks)
    assert "section" in out and '"table"' not in out
