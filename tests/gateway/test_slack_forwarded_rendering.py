"""Forward rendering preserves provenance, block order and one history/live body budget."""

import pytest

from tests.gateway import test_slack_forwarded_messages as fixtures

adapter = fixtures.adapter
_redirect_cache = fixtures._redirect_cache


@pytest.mark.asyncio
async def test_forward_block_order_survives_live_and_history(adapter):
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": "Deployment heading"}},
        fixtures._rich(fixtures._section(fixtures._text("Before the table"))),
        {"type": "section", "text": {"type": "mrkdwn", "text": "Table label"}},
        {"type": "table", "rows": [[{"type": "raw_text", "text": "Table contents"}]]},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": "Explanation below"}]},
    ]
    event = fixtures._event(attachments=[fixtures._share(
        message_blocks=[{"message": {"blocks": blocks}}])])
    delivered = await fixtures._delivered(adapter, event)
    for text in (delivered.text, adapter._render_message_text(event, "U_BOT")):
        positions = [text.index(part) for part in (
            "Deployment heading", "Before the table", "Table label", "Table contents", "Explanation below")]
        assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_quoted_forward_keeps_source_and_files_without_repeating_body(adapter):
    body = "the deploy failed twice on the staging box"
    event = fixtures._event(text=f"<@U_BOT> Please investigate: {body}", attachments=[fixtures._share(
        text=body, message_blocks=None, channel_name="operations",
        files=[{"id": "F_QUOTED", "name": "report.pdf", "mimetype": "application/pdf"}])])
    delivered = await fixtures._delivered(adapter, event)
    for text in (delivered.text, adapter._render_message_text(event, "U_BOT")):
        assert text.count(body) == 1
        assert "Forwarded Slack message from Scout in #operations" in text
        assert fixtures.PERMALINK in text
        assert "[file: report.pdf (application/pdf)]" in text
        assert "not available" not in text


@pytest.mark.asyncio
async def test_multiple_forward_bodies_share_budget_in_live_and_history(adapter):
    event = fixtures._event(attachments=[fixtures._share(
        text=letter * 6000, message_blocks=None, author_name=name)
        for letter, name in (("x", "Ana"), ("y", "Ben"), ("z", "Cal"))])
    delivered = await fixtures._delivered(adapter, event)
    history = adapter._render_message_text(event, "U_BOT")
    for text in (delivered.text, history):
        assert text.index("from Ana") < text.index("from Ben") < text.index("from Cal")
        assert text.count("x") + text.count("y") + text.count("z") <= 8000
        assert "omitted" in text
        assert text.startswith("look into this")
    assert delivered.text == history
