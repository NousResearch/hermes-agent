"""Streaming scrub of text-channel tool-call markup (#137775).

A model whose relay returned no native ``tool_calls`` serializes them as
``<tool_call>``/``<invoke>``/``<parameter>`` text. The whole-string
``strip_think_blocks`` knows that family, but the streaming path only ran
``StreamingThinkScrubber``, which bound to reasoning tags alone — so the markup
reached every stream consumer verbatim. These tests lock the scrubber's
tool-call handling: closed blocks (attribute-bearing, namespaced, split across
deltas), unterminated blocks dropped at flush, orphan closers, and prose
mentions preserved.
"""

from __future__ import annotations

import pytest

from agent.think_scrubber import TOOL_CALL_TAG_NAMES, StreamingThinkScrubber


def _drive(scrubber: StreamingThinkScrubber, deltas: list[str]) -> str:
    """Feed a sequence of deltas and return the concatenated visible output."""
    out = [scrubber.feed(d) for d in deltas]
    out.append(scrubber.flush())
    return "".join(out)


ISSUE_PAYLOAD = (
    "<tool_call>\n"
    '  <invoke name="update_plan">plan</invoke>\n'
    '  <invoke name="terminal"><parameter name="cmd">ls</parameter></invoke>\n'
    "</tool_call>\n"
    "Here is the listing."
)


class TestClosedToolCallBlocks:
    """Closed <tool_call>…</tool_call> blocks are always stripped, like reasoning pairs."""

    def test_issue_payload_split_across_deltas(self) -> None:
        s = StreamingThinkScrubber()
        deltas = [
            "<tool_call>\n",
            '  <invoke name="update_plan">',
            "plan</invoke>\n",
            '  <invoke name="terminal"><parameter name="cmd">ls</parameter></invoke>\n',
            "</tool_call>\nHere ",
            "is the listing.",
        ]
        assert _drive(s, deltas) == "\nHere is the listing."

    def test_closed_pair_single_delta(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["A<tool_call>secret</tool_call>B"]) == "AB"

    @pytest.mark.parametrize("tag", TOOL_CALL_TAG_NAMES)
    def test_all_tag_variants(self, tag: str) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, [f"<{tag}>x</{tag}>Hello"]) == "Hello"

    def test_attribute_bearing_open_tag(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ['<tool_call id="7">x</tool_call>done']) == "done"

    def test_namespaced_open_and_close(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<ns:function_calls>x</ns:function_calls>done"]) == "done"

    def test_case_insensitive(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<TOOL_CALL>x</TOOL_CALL>done"]) == "done"

    def test_char_by_char_stream(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, list(ISSUE_PAYLOAD)) == "\nHere is the listing."


class TestUnterminatedToolCallBlocks:
    """An unclosed <tool_call> is unrecoverable (#101899): drop the rest of the stream."""

    def test_unterminated_open_at_boundary_dropped_at_flush(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<tool_call>\n", '<invoke name="terminal">ls']) == ""

    def test_unterminated_open_after_prose(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["Working.\n<tool_call>", '<invoke name="x">']) == "Working.\n"


class TestPartialTagsAcrossDeltas:
    """Open tags split across deltas must be held back, not emitted raw."""

    def test_split_open_tag_held_back(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<tool_", "call>x</tool_call>done"]) == "done"

    def test_split_open_tag_with_attributes_held_back(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<tool_call ", 'id="7">x</tool_call>done']) == "done"

    def test_split_namespaced_open_tag_held_back(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<ns:tool_", "call>x</ns:tool_call>done"]) == "done"

    def test_split_close_tag_held_back(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<tool_call>x</tool_", "call>done"]) == "done"


class TestOrphanClosers:
    """Degenerate tails keep fragments like '</parameter></invoke>' with no opener."""

    def test_invoke_parameter_closers_stripped(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["</parameter>\n</invoke>\n", "prose"]) == "prose"

    def test_split_parameter_closer_held_back(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["</param", "eter>\nprose"]) == "prose"

    def test_tool_call_closer_alone_stripped(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["</tool_call>prose"]) == "prose"


class TestProseMentions:
    """Prose that merely mentions the tags must survive (boundary gating)."""

    def test_midline_mention_without_close_preserved(self) -> None:
        s = StreamingThinkScrubber()
        text = "Serialize the call as <tool_call> markup in your answer"
        assert _drive(s, [text]) == text


class TestAgreementWithWholeStringStrip:
    """The streaming and whole-string paths must agree on the same payload (#137775)."""

    def test_scrubber_matches_strip_think_blocks(self) -> None:
        from agent.agent_runtime_helpers import strip_think_blocks

        s = StreamingThinkScrubber()
        streamed = _drive(s, [ISSUE_PAYLOAD])
        settled = strip_think_blocks(None, ISSUE_PAYLOAD)
        assert streamed == settled == "\nHere is the listing."

    def test_shared_tag_name_list(self) -> None:
        from agent.agent_runtime_helpers import _TOOL_CALL_TAG_NAMES

        assert _TOOL_CALL_TAG_NAMES is TOOL_CALL_TAG_NAMES
