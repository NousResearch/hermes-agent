"""Nested ``attachments[].blocks[]`` text in link unfurls shares one bounded budget.

The live inbound path (``_append_link_unfurls``) renders each attachment's Block Kit
``blocks`` into the message the agent reads. Slack allows up to 20 attachments per message,
each carrying its own blocks, so without a shared ceiling the projection grows linearly with
attachment count while the top-level ``blocks`` path is capped once.
"""

from agent.compression_marker import _COMPRESSION_MARKER_RE
from plugins.platforms.slack.adapter import SlackAdapter


def _rich_blocks(sections: int = 8, size: int = 400) -> list:
    return [{"type": "rich_text", "elements": [
        {"type": "rich_text_section", "elements": [{"type": "text", "text": "x" * size}]}
        for _ in range(sections)]}]


def _attachment(i: int) -> dict:
    return {"title": f"Alert {i}", "title_link": "https://example.com/a", "blocks": _rich_blocks()}


def test_nested_block_text_stops_growing_with_attachment_count():
    sizes = {n: len(SlackAdapter._append_link_unfurls("intro", [_attachment(i) for i in range(n)]))
             for n in (1, 5, 20)}
    # Unbounded, 20 attachments cost 20x one (each ~3.2K of block text). Bounded, going from
    # 5 to 20 attachments adds only their headers, not another 15 bodies of block text.
    per_attachment_growth = (sizes[20] - sizes[5]) / 15
    assert per_attachment_growth < 200, sizes
    assert sizes[20] < 3 * sizes[1], sizes
    assert _COMPRESSION_MARKER_RE.search(SlackAdapter._append_link_unfurls("intro", [_attachment(i) for i in range(20)]))


def test_first_attachment_keeps_its_body_and_headers_survive_exhaustion():
    out = SlackAdapter._append_link_unfurls("intro", [_attachment(i) for i in range(20)])
    # The common single-alert shape is unchanged: attachment 0's block text is fully present.
    assert "x" * 400 in out
    # Later attachments still announce themselves even when the block budget is spent.
    assert "📎 [Alert 19](https://example.com/a)" in out
