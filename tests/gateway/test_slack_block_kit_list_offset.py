"""Outbound: an ordered list that resumes after other content keeps its numbering.

``render_blocks`` turns agent markdown into Slack ``rich_text_list`` elements.
Slack numbers each list from ``offset + 1``; without ``offset`` an agent reply
with "3." after a heading rendered in Slack as "1.". Inbound handling of
``offset`` is #76025.
"""

from plugins.platforms.slack.block_kit import render_blocks


def _lists(blocks):
    return [
        el for b in blocks if b["type"] == "rich_text"
        for el in b["elements"] if el["type"] == "rich_text_list"]


def test_list_resuming_after_heading_keeps_number():
    first, second = _lists(render_blocks("# A\n1. one\n2. two\n# B\n3. three\n4. four"))
    assert "offset" not in first
    assert second["offset"] == 2


def test_list_resuming_after_paragraph_keeps_number():
    first, second = _lists(render_blocks("1. one\n2. two\n\nSome prose.\n\n3. three"))
    assert "offset" not in first
    assert second["offset"] == 2


def test_list_starting_at_one_has_no_offset():
    (lst,) = _lists(render_blocks("1. one\n2. two"))
    assert "offset" not in lst


def test_bullets_never_get_offset():
    (lst,) = _lists(render_blocks("- a\n- b"))
    assert "offset" not in lst


def test_item_text_and_continuation_lines_unchanged():
    (lst,) = _lists(render_blocks("5. five\n   more five\n6. six"))
    assert lst["offset"] == 4
    texts = ["".join(e.get("text", "") for e in item["elements"]) for item in lst["elements"]]
    assert texts == ["five more five", "six"]
