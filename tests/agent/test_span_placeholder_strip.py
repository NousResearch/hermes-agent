"""Regression tests: <span_placeholder> tokens leaked into payloads — the
DeepSeek-V4.1 mm-span placeholder family (added-token ids 128847..129279 on
served checkpoint) reaches clients verbatim because the engine's chat path
keeps skip_special_tokens=False for tool-call parsing. The token's '|'
destroyed bash commands, file paths and even tool names (kanban t_46378669).
"""

import pytest

from agent.message_sanitization import (
    scrub_span_placeholders_from_messages,
    strip_span_placeholders,
)

FULL = "<|place_holder_mm_span_0442|>"
FULL_0179 = "<|place_holder_mm_span_0179|>"
TRUNC_PREFIX = "<|place_holder_mm_span_0442"  # cut before the trailing |>


class TestStripSpanPlaceholders:
    def test_full_token_removed(self):
        assert strip_span_placeholders(f"cd /tmp/agent1/m{FULL} 2>/dev/null") == "cd /tmp/agent1/m 2>/dev/null"

    def test_other_token_ids_removed(self):
        out = strip_span_placeholders(f"a {FULL_0179} b {FULL} c")
        assert "<|place" not in out and out == "a  b  c"

    def test_truncation_prefix_removed(self):
        assert strip_span_placeholders(f"cd /tmp/agent1/workspace{TRUNC_PREFIX}") == "cd /tmp/agent1/workspace"

    def test_short_prefixes_left_alone(self):
        # Sub-14-char debris is storage garbage, not live emission; the regex
        # deliberately refuses to match it to avoid eating innocent "<|p..." text.
        assert strip_span_placeholders("x <|pl y") == "x <|pl y"
        assert strip_span_placeholders("x <|place y") == "x <|place y"

    def test_clean_text_untouched_same_object(self):
        text = "normal prose with <|dsml|> markup intact"
        assert strip_span_placeholders(text) is text  # identity fast-path

    def test_non_string_passthrough(self):
        assert strip_span_placeholders(None) is None


class TestScrubSpanPlaceholdersFromMessages:
    def test_content_and_fields_scrubbed(self):
        msgs = [
            {"role": "assistant", "content": "Let me widen" + FULL + ": the update",
             "reasoning_content": "think" + FULL, "api_content": "sync" + FULL},
            {"role": "assistant", "tool_calls": [{"function": {
                "name": "read" + FULL + "_file",
                "arguments": '{"path": "/a/m' + FULL + '"}'}}]},
        ]
        assert scrub_span_placeholders_from_messages(msgs) == 2
        assert "<|place" not in "".join(str(m) for m in msgs)
        assert msgs[1]["tool_calls"][0]["function"]["name"] == "read_file"
        assert msgs[1]["tool_calls"][0]["function"]["arguments"] == '{"path": "/a/m"}'

    def test_ids_left_alone(self):
        # Call ids pair with tool-result rows; never touch them.
        msgs = [{"role": "assistant", "tool_calls": [{"id": f"call_x{FULL}", "function": {
            "name": "terminal", "arguments": "{}"}}]}]
        scrub_span_placeholders_from_messages(msgs)
        assert msgs[0]["tool_calls"][0]["id"] == f"call_x{FULL}"

    def test_multimodal_parts_scrubbed(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": f"see {FULL} here"}]}]
        assert scrub_span_placeholders_from_messages(msgs) == 1
        assert msgs[0]["content"][0]["text"] == "see  here"

    def test_clean_list_untouched_count_zero(self):
        msgs = [{"role": "user", "content": "clean"}, {"role": "assistant", "content": "also clean"}]
        assert scrub_span_placeholders_from_messages(msgs) == 0

    def test_non_list_no_crash(self):
        assert scrub_span_placeholders_from_messages(None) == 0  # type: ignore[arg-type]