"""Tests for StreamingToolCallScrubber (agent/tool_call_scrubber.py).

Contract, both ways: (1) for every covered tool-call XML shape the scrubber's chunk-split
output equals the final-response stripper's whole-string output — the stream is never
dirtier than the final answer. Where the final strips a line-anchored cut tail together
with the preceding ``newline + indentation``, the stream keeps that whitespace run (it was
already delivered before the anchor could be known) and suppresses the XML only; (2) no
delta consumer ever sees raw tool-call XML (the streaming repro from the review on
NousResearch/hermes-agent#114136). The scenarios map to the muse-spark / opencode-go
Responses-wire serialization: ``<atem:function_calls>…</atem:function_calls>`` blocks,
stray closers, and openers cut mid-serialization.
"""

from __future__ import annotations

from agent.agent_runtime_helpers import strip_think_blocks
from agent.tool_call_scrubber import StreamingToolCallScrubber

# The field block from the PR, and the cut-mid-serialization tail.
CLOSED_BLOCK = (
    "Checking the queue.\n"
    "<atem:function_calls>\n"
    '<atem:invoke name="default.terminal">\n'
    '<atem:parameter name="command">echo hi</atem:parameter>\n'
    "</atem:invoke>\n"
    "</atem:function_calls>"
)
CUT_TAIL = 'Waiting.\n<atem:function_calls>\n<atem:invoke name="default.terminal">'
REVIEWER_REPRO = (
    "redox=default.hermes_search_files Hollywood"
    "<atem:function_calls>...</atem:function_calls>"
)


def _drive(deltas: list[str]) -> str:
    """Feed a sequence of deltas and return the concatenated visible output."""
    s = StreamingToolCallScrubber()
    out = [s.feed(d) for d in deltas]
    out.append(s.flush())
    return "".join(out)


class TestCoveredShapes:
    def test_closed_block_single_delta(self):
        assert _drive([CLOSED_BLOCK]) == "Checking the queue.\n"

    def test_closed_block_split_across_deltas(self):
        deltas = [
            "Checking the queue.\n<atem:func",
            "tion_calls>\n<atem:invoke ",
            'name="x">y</atem:invoke></atem:func',
            "tion_calls>",
        ]
        assert _drive(deltas) == "Checking the queue.\n"

    def test_cut_mid_serialization_drops_the_line_anchored_tail(self):
        out = _drive([CUT_TAIL])
        assert "atem:" not in out
        assert out == "Waiting.\n"

    def test_stray_closer_is_suppressed(self):
        assert _drive(["done.</atem:function_calls> after"]) == "done.after"

    def test_plain_names_and_case_insensitivity(self):
        assert _drive(["a <Tool_Call>b</TOOL_CALL> c"]) == "a  c"

    def test_namespace_mismatch_between_open_and_close_still_closes(self):
        # the final pattern closes on any namespace prefix, keeping the space before '<'
        assert _drive(["x <tool_calls>y</atem:tool_calls>z"]) == "x z"

    def test_mid_line_open_without_closer_is_released_at_flush(self):
        # mirrors the final stripper exactly: no line anchor, no closer → kept
        assert _drive(["abc <tool_calls>xyz"]) == "abc <tool_calls>xyz"

    def test_prose_before_a_real_call_is_kept(self):
        assert _drive([REVIEWER_REPRO]) == "redox=default.hermes_search_files Hollywood"

    def test_non_tag_angle_text_survives(self):
        assert _drive(["a < b and 3 <4 stay"]) == "a < b and 3 <4 stay"


CLOSER_CUT = "Waiting.\n<atem:function_calls>{}\n</atem:function_"


class TestCutInsideCloser:
    """A stream ending inside an anchored block's closer is still the block (#114136)."""

    def test_pending_closer_tail_is_suppressed(self):
        out = _drive([CLOSER_CUT])
        assert "atem:" not in out
        assert "function_" not in out
        assert out == "Waiting.\n"

    def test_pending_closer_tail_is_suppressed_under_every_split(self):
        want = strip_think_blocks(None, CLOSER_CUT)
        for i in range(len(CLOSER_CUT) + 1):
            got = _drive([CLOSER_CUT[:i], CLOSER_CUT[i:]])
            assert "atem:" not in got, (i, got)
            assert got.rstrip() == want.rstrip(), (i, got, want)
        assert _drive(list(CLOSER_CUT)).rstrip() == want.rstrip()


NAMED_BLOCK = 'Hello\n<function name="search">query</function> done'


class TestNamedFunctionBlocks:
    """Named <function name=…> blocks are suppressed exactly where the final stripper
    removes them (boundary- and name-gated), whole or split across deltas."""

    def test_closed_block_single_delta(self):
        assert _drive([NAMED_BLOCK]) == "Hello\n done"

    def test_closed_block_split_across_deltas(self):
        deltas = [
            'Hello\n<function na',
            'me="search">qu',
            "ery</func",
            "tion> done",
        ]
        assert _drive(deltas) == "Hello\n done"

    def test_mid_line_prose_mention_keeps_text_drops_stray_closer(self):
        # No boundary: the final stripper keeps the opener and drops only </function>.
        assert _drive(['a <function name="x">b</function> c']) == 'a <function name="x">bc'

    def test_unterminated_block_is_released_like_the_final_stripper(self):
        text = 'Hello\n<function name="x">y done'
        assert _drive([text]) == text
        assert _drive(list(text)) == text

    def test_namespaced_closer_does_not_close(self):
        assert _drive(["X\n<function name=\"a\">b</ns:function>c</function>d"]) == "X\nd"


class TestArgKeyMarkup:
    """Line-anchored GLM <arg_key>/<arg_value> markup is suppressed through end of
    stream, mirroring the final stripper's drop of the line through end of text."""

    def test_line_anchored_markup_is_suppressed(self):
        out = _drive(["Hello\n<arg_key>name</arg_key> done"])
        assert "arg_key" not in out
        assert out == "Hello\n"

    def test_markup_split_across_deltas_is_suppressed(self):
        out = _drive(["Hello\n<arg_", "key>name</arg_", "value> done"])
        assert "arg_key" not in out
        assert "arg_value" not in out
        assert out == "Hello\n"

    def test_first_angle_on_line_gate(self):
        # A '<' earlier on the line keeps the text, exactly like the final stripper.
        text = "a < b <arg_key>x"
        assert _drive([text]) == text
        assert _drive(list(text)) == text


# Under every chunking the stream equals the final stripper byte for byte.
EXACT_CORPUS = [
    CLOSED_BLOCK,
    REVIEWER_REPRO,
    "abc <tool_calls>x</tool_calls> y",
    "x <tool_calls>y</tool_calls> z </tool_call> w",
    "mid</tool_calls>dle",
    "a < b and <tool_call>c</tool_call> d",
    "keep <this> and </that>",
    "<tool_calls foo='a>b'>x</tool_calls>y",
    "word<atem:funct",
    "plain text with no tags",
    "abc <tool_calls>xyz",
    "<TOOL_CALLS>x</tool_calls>after",
    "done.</atem:function_calls> after",
    NAMED_BLOCK,
    'end.<function name="x">y</function> done',
    '<function name="foo">bar</function> done',
    'a <function name="x">b</function> c',
    'Hello\n<function>bar</function> done',
    'Hello\n<function name="x">y done',
    'Hello\n<ns:function name="x">ydone',
    'X\n<function name="a">b</ns:function>c</function>d',
    'Hello\n<FUNCTION NAME="x">y</FUNCTION> done',
    "a < b <arg_key>x",
    "Hello\n<arg_k",
    "Hello\n<ns:arg_key>x",
    # A released mid-line block still loses its stray closers, like the final (#114136/08).
    "x <tool_calls>a</tool_result>b",
    "x <tool_calls>a</function>b",
    "x <tool_calls>a</atem:function_calls>b",
    "x <tool_calls>a</notatag>b",
]

# Line-anchored unresolved openers: the final also removes the preceding newline +
# indentation; the stream has already delivered that whitespace run, so it may differ
# there only (asserted via rstrip below) — and never in XML.
ANCHORED_CORPUS = [
    CUT_TAIL,
    "abc\n  <tool_calls>xyz",
    "one\n\n  <tool_call>a\nb",
    "   <tool_call>lead at stream start",
    CLOSER_CUT,
    "Hello\n<arg_key>name</arg_key> done",
    "<arg_key>x",
    'Hello\n<arg_key foo="bar">x',
    "Hello\n</arg_value> done",
    "Hello\n<ARG_KEY>x</ARG_KEY> done",
]


class TestParityWithFinalStripper:
    """The invariant the review asked for: chunk-split stream == final stripper output."""

    def test_exact_equality_under_every_single_split(self):
        for text in EXACT_CORPUS:
            want = strip_think_blocks(None, text)
            for i in range(len(text) + 1):
                got = _drive([text[:i], text[i:]])
                assert got == want, (text, i, got, want)

    def test_exact_equality_char_by_char(self):
        for text in EXACT_CORPUS:
            assert _drive(list(text)) == strip_think_blocks(None, text), text

    def test_anchored_cut_differs_only_in_leading_whitespace(self):
        for text in ANCHORED_CORPUS:
            want = strip_think_blocks(None, text)
            half = len(text) // 2
            for chunks in ([text], list(text), [text[:half], text[half:]]):
                got = _drive(chunks)
                assert got.rstrip() == want.rstrip(), (text, chunks, got, want)
                assert "atem:" not in got, (text, chunks, got)
                assert "arg_key" not in got and "arg_value" not in got, (text, chunks, got)

    def test_no_raw_markup_reaches_consumers(self):
        for text in (CLOSED_BLOCK, CUT_TAIL, REVIEWER_REPRO):
            out = _drive(list(text))
            assert "atem:" not in out
            assert "function_calls" not in out


class TestReleasedRawStrayClosers:
    """Stray closers inside a released mid-line block stay suppressed (#114136/08).

    The final stripper removes stray closers anywhere, independently of blocks; the
    mid-line raw released at flush() must do the same. Covered under every split by
    the EXACT_CORPUS entries above; these pin the headline specimens explicitly.
    """

    def test_stray_closers_of_other_names_are_suppressed(self):
        for text in (
            "x <tool_calls>a</tool_result>b",
            "x <tool_calls>a</function>b",
            "x <tool_calls>a</atem:function_calls>b",
        ):
            want = strip_think_blocks(None, text)
            assert want == "x <tool_calls>ab"
            assert _drive([text]) == want
            assert _drive(list(text)) == want

    def test_unrecognized_closer_in_released_raw_is_kept(self):
        text = "x <tool_calls>a</notatag>b"
        assert _drive([text]) == text
        assert _drive(list(text)) == text

    def test_anchored_block_with_stray_is_still_dropped_wholesale(self):
        text = "x\n<tool_calls>a</tool_result>b"
        want = strip_think_blocks(None, text)
        assert want == "x"
        for chunks in ([text], list(text)):
            got = _drive(chunks)
            assert got.rstrip() == want.rstrip(), (chunks, got, want)
            assert "tool_result" not in got


OVERFLOW_NEVER_CLOSES = "x\n<tool_calls " + "a" * 100 + ">y"
OVERFLOW_THEN_CLOSES = OVERFLOW_NEVER_CLOSES + "</tool_calls>z"
MIDLINE_OVERFLOW = "x <tool_calls " + "a" * 100 + ">y"


class TestCapOverflowAnchoredOpener:
    """A recognized line-anchored opener past the partial-tag cap stays held (#114136/09).

    The final strips from the anchor through end of text whether the tag completes
    later or never; the stream must hold the span in block mode instead of leaking
    the '<' as prose once the cap overflows. Mid-line overflow stays released: the
    final keeps it too.
    """

    def test_never_closing_overflow_matches_final(self):
        want = strip_think_blocks(None, OVERFLOW_NEVER_CLOSES)
        assert want == "x"
        for chunks in ([OVERFLOW_NEVER_CLOSES], list(OVERFLOW_NEVER_CLOSES)):
            got = _drive(chunks)
            assert got.rstrip() == want.rstrip(), (chunks, got, want)
            assert "<tool_calls" not in got

    def test_overflow_then_closer_suppresses_the_whole_block(self):
        want = strip_think_blocks(None, OVERFLOW_THEN_CLOSES)
        assert want == "x\nz"
        for chunks in ([OVERFLOW_THEN_CLOSES], list(OVERFLOW_THEN_CLOSES)):
            got = _drive(chunks)
            assert got == want, (chunks, got, want)

    def test_overflow_matches_final_under_every_single_split(self):
        for text in (OVERFLOW_NEVER_CLOSES, OVERFLOW_THEN_CLOSES):
            want = strip_think_blocks(None, text)
            for i in range(len(text) + 1):
                got = _drive([text[:i], text[i:]])
                assert got.rstrip() == want.rstrip(), (text, i, got, want)
                assert "<tool_calls" not in got, (text, i, got)

    def test_midline_overflow_is_still_released_like_the_final(self):
        want = strip_think_blocks(None, MIDLINE_OVERFLOW)
        assert want == MIDLINE_OVERFLOW
        assert _drive([MIDLINE_OVERFLOW]) == want
        assert _drive(list(MIDLINE_OVERFLOW)) == want


class TestUnicodeWhitespaceParity:
    """The stray-closer trailing-whitespace skip eats what the final eats (#114136/10)."""

    def test_nbsp_after_stray_closer_is_dropped(self):
        # literal NBSP (U+00A0) between the closer and 'b'
        text = "a</tool_calls>\u00a0b"
        assert strip_think_blocks(None, text) == "ab"
        assert _drive([text]) == "ab"
        assert _drive(list(text)) == "ab"

    def test_em_space_after_stray_closer_is_dropped(self):
        # literal EM SPACE (U+2003) between the closer and 'b'
        text = "a</tool_calls>\u2003b"
        assert strip_think_blocks(None, text) == "ab"
        assert _drive([text]) == "ab"
        assert _drive(list(text)) == "ab"

    def test_ascii_whitespace_skip_is_unchanged(self):
        assert _drive(["done.</atem:function_calls> after"]) == "done.after"


class TestTicket01ExtraRegressions:
    """CRLF and pending-'<' variants of the anchored pending-tail class (#114136/01)."""

    def test_crlf_pending_closer_tail_leaves_no_fragment(self):
        text = "x\r\n<tool_calls>y</tool_ca"
        want = strip_think_blocks(None, text)
        assert want == "x\r"
        for i in range(len(text) + 1):
            got = _drive([text[:i], text[i:]])
            assert "tool_ca" not in got, (i, got)
            assert got.rstrip() == want.rstrip(), (i, got, want)
        assert _drive(list(text)).rstrip() == want.rstrip()

    def test_pending_lt_inside_anchored_block_leaves_no_fragment(self):
        text = "x\n<tool_calls>ab<"
        want = strip_think_blocks(None, text)
        assert want == "x"
        for i in range(len(text) + 1):
            got = _drive([text[:i], text[i:]])
            assert "<" not in got, (i, got)
            assert got.rstrip() == want.rstrip(), (i, got, want)
        assert _drive(list(text)).rstrip() == want.rstrip()
