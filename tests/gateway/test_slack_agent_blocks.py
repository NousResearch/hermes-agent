"""Unit tests for agent-authored Slack blocks (``slack.extra.agent_blocks``; pure function, no adapter)."""

import json

from plugins.platforms.slack.block_kit import MAX_BLOCKS, extract_agent_blocks

LAYOUT = {"type": "layout", "rows": [{"items": [{"block": {"type": "section", "text": {"type": "mrkdwn", "text": "*Paying*\n9"}}}]}],
          "fallback": [{"type": "section", "text": {"type": "mrkdwn", "text": "Paying: 9"}}]}
HEADER = {"type": "header", "text": {"type": "plain_text", "text": "Weekly metrics"}}


def fenced(payload, fence="```"):
    return f"{fence}slack-blocks\n{json.dumps(payload)}\n{fence}"


class TestExtractAgentBlocks:
    def test_plain_reply_is_not_claimed(self):
        assert extract_agent_blocks("") is None
        assert extract_agent_blocks("Here are the numbers: 9 paying.") is None
        assert extract_agent_blocks("```json\n[]\n```") is None

    def test_list_payload_with_surrounding_text(self):
        text, blocks = extract_agent_blocks("Here you go.\n\n" + fenced([HEADER, LAYOUT]) + "\n")
        assert text == "Here you go."
        assert [b["type"] for b in blocks] == ["header", "layout"]   # new block types pass through

    def test_object_payload_text_is_the_fallback(self):
        text, blocks = extract_agent_blocks(fenced({"text": "Weekly metrics: 9 paying", "blocks": [HEADER]}))
        assert text == "Weekly metrics: 9 paying"
        assert blocks == [HEADER]

    def test_no_text_anywhere_uses_the_header(self):
        text, _ = extract_agent_blocks(fenced([HEADER]))
        assert text == "Weekly metrics"

    def test_tilde_fence_and_longer_backticks(self):
        assert extract_agent_blocks(fenced([HEADER], "~~~")) is not None
        assert extract_agent_blocks(fenced([HEADER], "````")) is not None

    def test_invalid_payloads_fall_back_to_plain_text(self):
        assert extract_agent_blocks("```slack-blocks\n[{oops\n```") is None          # not JSON
        assert extract_agent_blocks(fenced({"blocks": "nope"})) is None              # not a list
        assert extract_agent_blocks(fenced([{"type": "section", "text": {"type": "mrkdwn", "text": ""}}])) is None
        assert extract_agent_blocks(fenced([HEADER]) + "\n" + fenced([HEADER])) is None   # two payloads

    def test_unclosed_fence_is_not_claimed(self):
        assert extract_agent_blocks("```slack-blocks\n" + json.dumps([HEADER])) is None

    def test_many_unclosed_openers_stay_linear(self):
        import time
        content = "```slack-blocks\nx\n" * 20000          # ~400 KB of openers, none closed
        started = time.monotonic()
        assert extract_agent_blocks(content) is None
        assert time.monotonic() - started < 1.0

    def test_fence_inside_text_lines(self):
        text, blocks = extract_agent_blocks("Intro\n" + fenced([HEADER]) + "\nOutro")
        assert text == "Intro\nOutro"
        assert blocks == [HEADER]

    def test_broadcasts_are_made_inert(self):
        section = {"type": "section", "text": {"type": "mrkdwn", "text": "ping <!channel> and <!here|here>"}}
        rich = {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": [
            {"type": "broadcast", "range": "everyone"}, {"type": "text", "text": " hi"}]}]}
        text, blocks = extract_agent_blocks(fenced({"text": "<!everyone> look", "blocks": [section, rich]}))
        assert "<!" not in json.dumps(blocks) and "<!" not in text
        assert blocks[0]["text"]["text"] == "ping &lt;!channel> and &lt;!here|here>"
        assert blocks[1]["elements"][0]["elements"][0] == {"type": "text", "text": "@everyone"}

    def test_payload_is_sanitized(self):
        table = {"type": "table", "rows": [[{"type": "raw_text", "text": "a"}]], "column_settings": [None, {"align": "right"}]}
        _, blocks = extract_agent_blocks(fenced([HEADER] * (MAX_BLOCKS + 5) + [table]))
        assert len(blocks) == MAX_BLOCKS
        _, (fixed,) = extract_agent_blocks(fenced([table]))
        assert fixed["column_settings"] == [{}, {"align": "right"}]
