"""Tests for StreamingToolCallFragmentScrubber.

The scrubber removes leaked tool-call XML opener fragments that Qwen3 and
similar models emit as plain text deltas before switching to native
tool_calls.  The key correctness properties verified here are:

  1. **Prose context (in_toolcall_context=False) is pure passthrough** —
     no stripping of any kind occurs.  This is the F1 fix: short suffixes
     like 'call>', 'all>', 'll>', 'l>', 's>' are substrings of ordinary
     HTML tags and prose words; stripping them unconditionally corrupts
     '<ul>', '<ol>', '<html>', '<small>', etc.

  2. **Tool-call context (in_toolcall_context=True)**: every opener-suffix
     fragment with len > 1 is stripped, including short ones like 'l_call>',
     '_call>', 'call>'.  The lone '>' is NOT stripped in either context.

  3. Prose that precedes a fragment in TC context is preserved.

  4. Case-insensitivity in TC context.

  5. Split-delta regressions in TC context: fragments assembled across two
     or three deltas are scrubbed to empty.

  6. Context-aware preservation:
       - Mid-word suffixes (no leading '<') are stripped ONLY in TC context.
       - Leading-'<' openers are preserved in prose context (default) and
         stripped in tool-call context (in_toolcall_context=True).
       - Split deltas that reconstruct a leading-'<' opener follow the
         same context rule as a single-delta leading-'<' opener.

  7. '>' in prose round-trips unchanged: 'a > b', '> quote', '->', etc.

  8. flush() emits any held innocent tail verbatim — no data loss.

  9. Empty / no-op passthrough.

  10. F1 regression lockdown: HTML tags and prose words that happen to
      contain fragment suffixes pass through unchanged in prose context.

  11. F2 documented limitation: a leaked opener prefix split across the
      Path A → Path B boundary may not be fully scrubbed (accepted edge
      case).
"""

import pytest

from agent.toolcall_fragment_scrubber import StreamingToolCallFragmentScrubber


# ── Helper ───────────────────────────────────────────────────────────────


def _drive(
    scrubber: StreamingToolCallFragmentScrubber,
    deltas: list[str],
    in_toolcall_context: bool = False,
) -> str:
    """Feed each delta through *scrubber* and append flush().

    Resets the scrubber before driving so tests are independent of call order.
    All deltas use the same in_toolcall_context value.
    """
    scrubber.reset()
    parts: list[str] = []
    for delta in deltas:
        result = scrubber.feed(delta, in_toolcall_context=in_toolcall_context)
        parts.append(result)
    tail = scrubber.flush()
    parts.append(tail)
    return "".join(parts)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def s() -> StreamingToolCallFragmentScrubber:
    return StreamingToolCallFragmentScrubber()


# ── Single-delta fragments in TC context (stripped) ───────────────────────
# Note: the lone '>' is NOT in this list — it must NOT be stripped.
# These tests all use in_toolcall_context=True (the only context where stripping occurs).


class TestSingleDeltaFragmentsInTcContext:
    """Every opener-suffix fragment with len > 1 in TC context becomes empty.

    The lone '>' is excluded — see TestGtInProse for coverage of that.
    Mid-word suffixes in PROSE context are passthrough (see TestF1HtmlProseLockdown).
    """

    @pytest.mark.parametrize("fragment", [
        # <tool_call> suffixes (len > 1 only) — all stripped in TC ctx
        "<tool_call>",
        "tool_call>",
        "ool_call>",
        "ol_call>",
        "l_call>",
        "_call>",
        "call>",
        "all>",
        "ll>",
        "l>",
        # <tool_calls> suffixes (unique to this opener, len > 1)
        "<tool_calls>",
        "tool_calls>",
        "ool_calls>",
        "ol_calls>",
        "l_calls>",
        "_calls>",
        "calls>",
        "alls>",
        "lls>",
        "ls>",
        "s>",
        # <function_call> suffixes (len > 1)
        "<function_call>",
        "function_call>",
        "unction_call>",
        "nction_call>",
        "ction_call>",
        "tion_call>",
        "ion_call>",
        "on_call>",
        "n_call>",
        # <function_calls> suffixes (unique tail, len > 1)
        "<function_calls>",
        "function_calls>",
        "calls>",
    ])
    def test_fragment_stripped_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber, fragment: str
    ) -> None:
        """Every opener-suffix fragment is stripped in tool-call context."""
        assert _drive(s, [fragment], in_toolcall_context=True) == "", (
            f"Expected empty for fragment {fragment!r} in TC context"
        )

    def test_all_tool_call_suffixes_stripped_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Exhaustive check: all positional suffixes of <tool_call> with len>1 stripped in TC."""
        opener = "<tool_call>"
        for start in range(len(opener)):
            fragment = opener[start:]
            if len(fragment) <= 1:
                continue  # lone '>' is not stripped by design
            result = _drive(s, [fragment], in_toolcall_context=True)
            assert result == "", (
                f"suffix at position {start} ({fragment!r}) not stripped in TC ctx"
            )

    def test_all_function_call_suffixes_stripped_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Exhaustive check: all positional suffixes of <function_call> with len>1 stripped in TC."""
        opener = "<function_call>"
        for start in range(len(opener)):
            fragment = opener[start:]
            if len(fragment) <= 1:
                continue  # lone '>' is not stripped by design
            result = _drive(s, [fragment], in_toolcall_context=True)
            assert result == "", (
                f"suffix at position {start} ({fragment!r}) not stripped in TC ctx"
            )

    def test_lone_gt_is_NOT_stripped(self, s: StreamingToolCallFragmentScrubber) -> None:
        """The lone '>' must pass through unchanged in both contexts.

        Stripping it would corrupt 'a > b', '> quote', '->', HTML, etc.
        """
        assert _drive(s, [">"], in_toolcall_context=False) == ">", (
            "Bare '>' must not be stripped in prose ctx"
        )
        assert _drive(s, [">"], in_toolcall_context=True) == ">", (
            "Bare '>' must not be stripped in TC ctx"
        )


# ── F1 regression lockdown: prose context is pure passthrough ─────────────


class TestF1HtmlProseLockdown:
    """Prose context (in_toolcall_context=False) must be byte-for-byte passthrough.

    This is the F1 fix lockdown.  The OLD implementation stripped short
    suffixes like 'call>', 'll>', 'l>', 's>' unconditionally, which
    corrupted HTML tags and prose words:
      <ul>    → <u      (l> stripped)
      <ol>    → <o      (l> stripped)
      <html>  → <htm    (l> stripped)
      <small> → <sm     (all> stripped)
      <rules> → <rule   (s> stripped)
      <files> → <file   (s> stripped)
      balls>  → b       (all> stripped)

    After the fix, ALL of these must round-trip unchanged in prose context.
    """

    @pytest.mark.parametrize("text", [
        # HTML tags that contain opener suffix substrings
        "<ul>",
        "<ol>",
        "<html>",
        "<small>",
        "<details>",
        "<rules>",
        "<files>",
        # Prose words ending with fragment-like suffixes
        "balls>",
        "calls>",
        "I have 3 balls> here",
        # Legitimate HTML markup
        "<ul><li>a</li></ul>",
        "<ol><li>item</li></ol>",
        # Comparison operators
        "a > b",
        "-> arrow",
        "x => y",
        # Angle bracket non-openers
        "price < 10",
        # Full tool-call opener used as documentation
        "<tool_call>",
        "<tool_call>x</tool_call>",
        # Mid-word suffixes that are plain text in prose
        "ool_call>",
        "l_call>",
        "_call>",
        "call>",
        "all>",
        "ll>",
        "l>",
        "s>",
    ])
    def test_prose_passthrough(
        self, s: StreamingToolCallFragmentScrubber, text: str
    ) -> None:
        """Every text input passes through unchanged in prose context (in_toolcall_context=False)."""
        result = _drive(s, [text], in_toolcall_context=False)
        assert result == text, (
            f"Prose passthrough broken: input={text!r} output={result!r}"
        )

    def test_multi_delta_prose_passthrough(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Multi-delta prose stream is passthrough: no held bytes, no stripping."""
        s.reset()
        parts = []
        for delta in ["<ul>", "<li>", "item", "</li>", "</ul>"]:
            parts.append(s.feed(delta, in_toolcall_context=False))
        parts.append(s.flush())
        assert "".join(parts) == "<ul><li>item</li></ul>"

    def test_prose_with_trailing_ool_call_passthrough(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """'hello ool_call>' in prose context passes through unchanged (F1 fix)."""
        result = _drive(s, ["hello ool_call>"], in_toolcall_context=False)
        assert result == "hello ool_call>", (
            f"F1 regression: prose got corrupted: {result!r}"
        )

    def test_prose_split_deltas_no_hold(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """In prose context, no bytes are held between deltas — each fires immediately."""
        s.reset()
        r1 = s.feed("<too", in_toolcall_context=False)
        # In prose context '<too' is NOT held — it is immediately returned.
        assert r1 == "<too", (
            f"Prose ctx should not hold bytes: got {r1!r}"
        )
        r2 = s.feed("l_call>", in_toolcall_context=False)
        assert r2 == "l_call>", (
            f"Prose ctx should return unchanged: got {r2!r}"
        )
        tail = s.flush()
        assert r1 + r2 + tail == "<tool_call>", (
            "Prose split deltas should concatenate byte-for-byte"
        )


# ── '>' in prose regression tests (CRITICAL 1 lockdown) ──────────────────


class TestGtInProse:
    """'>' in normal prose must round-trip byte-for-byte.

    These tests lock in the fix for CRITICAL 1: _build_opener_suffixes
    now excludes single-character suffixes so bare '>' is never matched.
    """

    @pytest.mark.parametrize("text", [
        "a > b",
        "> quote",
        "-> arrow",
        "x => y",
        "a > b > c",
    ])
    def test_gt_in_prose_roundtrips(
        self, s: StreamingToolCallFragmentScrubber, text: str
    ) -> None:
        """Plain text containing '>' is emitted unchanged."""
        assert _drive(s, [text]) == text, f"'>' corrupted in {text!r}"

    def test_html_not_corrupted(self, s: StreamingToolCallFragmentScrubber) -> None:
        """HTML tags that are not tool-call openers must pass through unchanged."""
        assert _drive(s, ["<p>text</p>"]) == "<p>text</p>"

    def test_template_tag_not_corrupted(self, s: StreamingToolCallFragmentScrubber) -> None:
        """A '<template>' tag (not a recognised opener) must pass through unchanged."""
        assert _drive(s, ["<template>"]) == "<template>"


# ── Prose preservation (TC context stripping) ─────────────────────────────


class TestProsePreservation:
    """In TC context, prose before a fragment is preserved; only the fragment is removed.

    In PROSE context all text passes through unchanged (see TestF1HtmlProseLockdown).
    """

    def test_prose_plus_trailing_fragment_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Prose + trailing mid-word fragment in TC context: prose kept, fragment dropped."""
        assert _drive(s, ["hello ool_call>"], in_toolcall_context=True) == "hello "

    def test_prose_plus_tool_call_opener_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        # In TC context, the opener is stripped; prose before it is kept.
        assert _drive(s, ["Let me check <tool_call>"], in_toolcall_context=True) == "Let me check "

    def test_prose_plus_tool_call_opener_in_prose_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        # In prose context, the full <tool_call> is preserved (passthrough).
        assert _drive(s, ["Let me check <tool_call>"]) == "Let me check <tool_call>"

    def test_prose_plus_function_call_opener_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        assert _drive(s, ["calling <function_call>"], in_toolcall_context=True) == "calling "

    def test_prose_plus_short_fragment_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        assert _drive(s, ["text l_call>"], in_toolcall_context=True) == "text "

    def test_prose_plus_underscore_fragment_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        assert _drive(s, ["response _call>"], in_toolcall_context=True) == "response "

    def test_plain_prose_untouched(self, s: StreamingToolCallFragmentScrubber) -> None:
        text = "Here is a tool call for you."
        assert _drive(s, [text]) == text

    def test_prose_with_tool_call_mention(self, s: StreamingToolCallFragmentScrubber) -> None:
        text = "I will use the tool_call mechanism to query the API."
        assert _drive(s, [text]) == text

    def test_no_tag_passthrough(self, s: StreamingToolCallFragmentScrubber) -> None:
        text = "Hello, world! No tags here."
        assert _drive(s, [text]) == text

    def test_empty_string_passthrough(self, s: StreamingToolCallFragmentScrubber) -> None:
        assert _drive(s, [""]) == ""

    def test_whitespace_only_passthrough(self, s: StreamingToolCallFragmentScrubber) -> None:
        assert _drive(s, ["   \n  "]) == "   \n  "


# ── Case-insensitivity (TC context) ──────────────────────────────────────


class TestCaseInsensitivity:
    """Fragments are stripped regardless of case in TC context."""

    @pytest.mark.parametrize("fragment", [
        "<TOOL_CALL>",
        "TOOL_CALL>",
        "OOL_CALL>",
        "L_CALL>",
        "<Tool_Call>",
        "<FUNCTION_CALL>",
        "Function_Call>",
        "FUNCTION_CALLS>",
    ])
    def test_uppercase_stripped_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber, fragment: str
    ) -> None:
        assert _drive(s, [fragment], in_toolcall_context=True) == "", (
            f"Expected empty for {fragment!r} in TC ctx"
        )

    def test_mixed_case_prose_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        assert _drive(s, ["hello OOL_CALL>"], in_toolcall_context=True) == "hello "


# ── Split-delta regressions (TC context) ─────────────────────────────────


class TestSplitDeltaRegressions:
    """Fragments split across multiple deltas are still scrubbed in TC context.

    All tests here use in_toolcall_context=True.
    For prose-context split behavior, see TestF1HtmlProseLockdown.
    """

    def test_split_too_ol_call_tc_ctx(self, s: StreamingToolCallFragmentScrubber) -> None:
        """'<too' + 'l_call>' in TC ctx → stripped."""
        assert _drive(s, ["<too", "l_call>"], in_toolcall_context=True) == ""

    def test_split_to_ol_call_tc_ctx(self, s: StreamingToolCallFragmentScrubber) -> None:
        """'<to' + 'ol_call>' in TC ctx → stripped."""
        assert _drive(s, ["<to", "ol_call>"], in_toolcall_context=True) == ""

    def test_split_t_ool_call_tc_ctx(self, s: StreamingToolCallFragmentScrubber) -> None:
        """'<t' + 'ool_call>' in TC ctx → stripped."""
        assert _drive(s, ["<t", "ool_call>"], in_toolcall_context=True) == ""

    def test_split_angle_only_tc_ctx(self, s: StreamingToolCallFragmentScrubber) -> None:
        """'<' + 'tool_call>' in TC ctx → stripped."""
        assert _drive(s, ["<", "tool_call>"], in_toolcall_context=True) == ""

    def test_split_prose_fragment_more_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Prose before split, fragment in mid-delta, more prose after (TC ctx)."""
        assert _drive(s, ["text <t", "ool_call> more"], in_toolcall_context=True) == "text  more"

    def test_three_way_split_tc_ctx(self, s: StreamingToolCallFragmentScrubber) -> None:
        """'<', 'tool', '_call>' assembled across three deltas → stripped (TC ctx)."""
        assert _drive(s, ["<", "tool", "_call>"], in_toolcall_context=True) == ""

    def test_split_function_call(self, s: StreamingToolCallFragmentScrubber) -> None:
        """Function call fragment split across deltas — TC context.

        '<func' + 'tion_call>' assembles to '<function_call>' which starts
        with '<'.  In TC context this is stripped.  In prose context it is
        preserved (see test_split_mid_word_suffix_preserved_in_prose_ctx).
        """
        assert _drive(s, ["<func", "tion_call>"], in_toolcall_context=True) == ""

    def test_split_function_calls(self, s: StreamingToolCallFragmentScrubber) -> None:
        """<function_calls> fragment split — TC context."""
        assert _drive(s, ["<function_call", "s>"], in_toolcall_context=True) == ""

    def test_split_tool_calls(self, s: StreamingToolCallFragmentScrubber) -> None:
        """<tool_calls> fragment split — TC context."""
        assert _drive(s, ["<tool_call", "s>"], in_toolcall_context=True) == ""

    def test_no_cross_contamination_between_turns(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """reset() clears held state so a prior split doesn't affect a new turn."""
        s.reset()
        s.feed("<to", in_toolcall_context=True)
        # Second turn reset — the held '<to' must not pollute this turn
        s.reset()
        result = s.feed("normal text")
        result += s.flush()
        assert result == "normal text"

    def test_split_mid_word_suffix_preserved_in_prose_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Split leading-'<' opener in prose context is pure passthrough.

        '<func' + 'tion_call>' in prose context:
        '<func' is NOT held (prose = passthrough), so it is emitted immediately.
        'tion_call>' is also emitted immediately.
        Combined result: '<function_call>'.
        """
        s.reset()
        r1 = s.feed("<func", in_toolcall_context=False)
        r2 = s.feed("tion_call>", in_toolcall_context=False)
        tail = s.flush()
        result = r1 + r2 + tail
        assert result == "<function_call>", (
            f"Prose ctx passthrough for split opener should give '<function_call>', got {result!r}"
        )

    def test_split_mid_word_suffix_stripped_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Same split in TC context → stripped."""
        result = _drive(s, ["<func", "tion_call>"], in_toolcall_context=True)
        assert result == "", (
            f"Split leading-'<' opener in TC ctx should be stripped, got {result!r}"
        )


# ── Context-aware preservation ────────────────────────────────────────────


class TestContextAwareStripping:
    """The scrubber must distinguish prose context from tool-call context.

    F1 fix: stripping only occurs in TC context (in_toolcall_context=True).
    Prose context (False) is pure passthrough — no mid-word suffix stripping.

    Preservation rules (see module docstring):
      - Prose context: pure passthrough, ALL text returned unchanged.
      - TC context mid-word suffixes (no '<'): always stripped.
      - TC context leading-'<' openers: stripped in TC context.
      - Split deltas resolving to leading-'<' opener: same rule.
    """

    def test_leading_opener_preserved_in_prose_context(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """A bare '<tool_call>' in prose context is preserved verbatim (passthrough)."""
        assert _drive(s, ["<tool_call>"]) == "<tool_call>"

    def test_leading_opener_stripped_in_tc_context(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """A bare '<tool_call>' in TC context is stripped (leaked boilerplate)."""
        assert _drive(s, ["<tool_call>"], in_toolcall_context=True) == ""

    def test_midword_suffix_passthrough_in_prose(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """'ool_call>' in prose context passes through unchanged (F1 fix — INVERTED)."""
        assert _drive(s, ["ool_call>"]) == "ool_call>"

    def test_midword_suffix_stripped_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """'ool_call>' is stripped in TC context."""
        assert _drive(s, ["ool_call>"], in_toolcall_context=True) == ""

    def test_split_delta_preserved_in_prose_context(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Split-delta '<tool_call>' + 'content</tool_call>' in prose is preserved.

        In prose context, '<tool_call>' is returned immediately (passthrough —
        no hold).  The subsequent 'content</tool_call>' is also returned unchanged.
        """
        s.reset()
        r1 = s.feed("<tool_call>")
        r2 = s.feed("content</tool_call>")
        r3 = s.flush()
        result = r1 + r2 + r3
        assert result == "<tool_call>content</tool_call>", (
            f"Split-delta prose <tool_call> must be preserved, got {result!r}"
        )

    def test_split_delta_opener_stripped_in_tc_context(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Split '<too' + 'l_call>' in TC context → stripped."""
        assert _drive(s, ["<too", "l_call>"], in_toolcall_context=True) == ""

    def test_full_balanced_xml_in_prose_preserved(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """A complete <tool_call>...</tool_call> in one chunk, prose context, is preserved."""
        text = "<tool_call>some content here</tool_call>"
        assert _drive(s, [text]) == text

    def test_full_balanced_xml_in_tc_ctx_strips_opener(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """In TC context, the opener part is stripped but close tag is protected."""
        text = "<tool_call>some content here</tool_call>"
        result = _drive(s, [text], in_toolcall_context=True)
        # The opener is stripped but the content and close tag survive.
        assert "<tool_call>" not in result
        assert "some content here" in result
        assert "</tool_call>" in result


# ── Preservation: complete XML constructs (prose context) ────────────────


class TestPreservation:
    """Complete/balanced XML constructs in prose must not be corrupted."""

    def test_balanced_function_call_tags(self, s: StreamingToolCallFragmentScrubber) -> None:
        text = "<function_call>args here</function_call>"
        assert _drive(s, [text]) == text

    def test_balanced_tags_in_prose(self, s: StreamingToolCallFragmentScrubber) -> None:
        text = "See the <tool_call>name</tool_call> example."
        assert _drive(s, [text]) == text

    def test_tool_call_opener_followed_by_close(self, s: StreamingToolCallFragmentScrubber) -> None:
        """A lone opener immediately followed by close tag is preserved in prose."""
        text = "<tool_call></tool_call>"
        assert _drive(s, [text]) == text

    def test_plain_prose_mentioning_tool_call(self, s: StreamingToolCallFragmentScrubber) -> None:
        text = "Use the tool_call API to invoke functions."
        assert _drive(s, [text]) == text

    def test_prose_with_angle_bracket_not_opener(self, s: StreamingToolCallFragmentScrubber) -> None:
        """'<' followed by non-opener text is emitted verbatim on flush."""
        result = _drive(s, ["price < 10"])
        assert result == "price < 10"


# ── flush() correctness ───────────────────────────────────────────────────


class TestFlushBehaviour:
    """flush() must emit innocent held tails verbatim, never silently drop them."""

    def test_innocent_partial_emitted_on_flush_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """'<to' held mid-stream in TC ctx, but stream ends — emitted verbatim."""
        s.reset()
        out = s.feed("<to", in_toolcall_context=True)
        assert out == ""  # held, not emitted yet
        tail = s.flush()
        assert tail == "<to"

    def test_prose_ctx_no_hold_so_flush_is_empty(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """In prose ctx '<to' is NOT held — returned immediately; flush() returns ''."""
        s.reset()
        out = s.feed("<to", in_toolcall_context=False)
        assert out == "<to", "Prose ctx should return '<to' immediately (no hold)"
        tail = s.flush()
        assert tail == "", "flush() has nothing held after prose ctx feed"

    def test_single_angle_bracket_held_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        s.reset()
        out = s.feed("<", in_toolcall_context=True)
        assert out == ""
        tail = s.flush()
        assert tail == "<"

    def test_single_angle_bracket_passthrough_in_prose(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        s.reset()
        out = s.feed("<", in_toolcall_context=False)
        assert out == "<"
        tail = s.flush()
        assert tail == ""

    def test_empty_scrubber_flush(self, s: StreamingToolCallFragmentScrubber) -> None:
        s.reset()
        assert s.flush() == ""

    def test_complete_fragment_on_flush_is_suppressed_in_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """A complete fragment held until flush() is suppressed in TC context."""
        s.reset()
        out1 = s.feed("<tool", in_toolcall_context=True)
        out2 = s.feed("_call>", in_toolcall_context=True)
        tail = s.flush()
        assert out1 + out2 + tail == ""

    def test_complete_fragment_on_flush_preserved_in_prose_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """A '<tool_call>' in prose context is returned immediately (not held); flush() is empty."""
        s.reset()
        # Feed '<tool_call>' in prose context (default) — passthrough, not held
        out = s.feed("<tool_call>")
        tail = s.flush()
        result = out + tail
        # The opener starts with '<' and we're in prose ctx → passthrough.
        assert result == "<tool_call>", (
            f"Prose-ctx opener passthrough on flush, got {result!r}"
        )

    def test_reset_clears_held_buffer(self, s: StreamingToolCallFragmentScrubber) -> None:
        s.reset()
        s.feed("<t", in_toolcall_context=True)
        s.reset()
        assert s._buf == ""
        assert s.flush() == ""

    def test_empty_delta_in_prose_flushes_held_buf(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Empty delta in prose context must flush any held TC-context buffer.

        Regression for the early-guard data-loss bug: feed('<to', True) holds
        '<to' in _buf; feed('', False) must flush it rather than short-circuit.
        Without the fix, the empty-text guard returned '' before reaching the
        Path A flush block, silently dropping the held bytes.
        """
        s.reset()
        out1 = s.feed("<to", in_toolcall_context=True)
        assert out1 == "", "TC ctx should hold '<to'"
        assert s._buf == "<to", "Buffer should be holding '<to'"
        out2 = s.feed("", in_toolcall_context=False)
        assert out2 == "<to", "Prose empty delta must flush held buffer '<to'"
        assert s._buf == "", "_buf must be cleared after prose flush"

    def test_empty_delta_in_tc_ctx_retains_buf(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Empty delta in TC context must NOT flush the held buffer (still mid-fragment)."""
        s.reset()
        out1 = s.feed("<to", in_toolcall_context=True)
        assert out1 == "", "TC ctx holds '<to'"
        out2 = s.feed("", in_toolcall_context=True)
        assert out2 == "", "Empty TC delta is a no-op"
        assert s._buf == "<to", "_buf must still hold '<to' after empty TC delta"
        tail = s.flush()
        assert tail == "<to", "flush() should emit the innocent partial '<to'"


# ── Multi-delta realistic scenarios ──────────────────────────────────────


class TestRealisticScenarios:
    """Realistic multi-delta patterns modelled on actual Qwen3 output."""

    def test_qwen3_text_then_fragment_then_toolcall(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Simulate: text delta, fragment delta (TC ctx), then native tool_calls follow."""
        result = _drive(s, [
            "I will look up that for you.",
            "<tool_call>",
        ], in_toolcall_context=True)
        assert result == "I will look up that for you."

    def test_qwen3_split_fragment_mid_text(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Text + split opener + text — only the opener is removed (TC ctx)."""
        result = _drive(s, [
            "Processing ",
            "<tool",
            "_call>",
            " done.",
        ], in_toolcall_context=True)
        assert result == "Processing  done."

    def test_multiple_fragments_across_turns(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Each turn (reset) is independent.  TC context fragments are stripped."""
        r1 = _drive(s, ["ool_call>"], in_toolcall_context=True)
        r2 = _drive(s, ["l_call>"], in_toolcall_context=True)
        r3 = _drive(s, ["normal response"])
        assert r1 == ""
        assert r2 == ""
        assert r3 == "normal response"

    def test_chars_one_by_one_tc_ctx(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Character-by-character feeding of '<tool_call>' in TC ctx produces ''."""
        fragment = "<tool_call>"
        result = _drive(s, list(fragment), in_toolcall_context=True)
        assert result == ""

    def test_no_tag_chars_by_char_passthrough(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Character-by-character feeding of normal text is byte-for-byte identical."""
        text = "Hello world"
        result = _drive(s, list(text))
        assert result == text

    def test_midword_suffix_stripped_in_tc_ctx_only(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Mid-word suffixes are stripped ONLY in TC context (F1 fix — INVERTED from old behavior)."""
        for frag in ["ool_call>", "l_call>", "_call>", "call>", "tool_call>", "l>", "s>"]:
            # In TC context: stripped
            result_tc = _drive(s, [frag], in_toolcall_context=True)
            assert result_tc == "", (
                f"Mid-word suffix {frag!r} should be stripped in TC ctx"
            )
            # In prose context: passthrough (F1 fix)
            result_prose = _drive(s, [frag], in_toolcall_context=False)
            assert result_prose == frag, (
                f"Mid-word suffix {frag!r} should pass through in prose ctx (F1 fix)"
            )


# ── F2 documented-limitation test ────────────────────────────────────────


class TestF2BoundaryLimitation:
    """Document and lock in the A→B boundary split limitation (F2).

    When a leaked opener's PREFIX arrives on Path A (prose, passthrough)
    immediately before the tool-call transition, and the SUFFIX arrives on
    Path B (TC context), the prefix has already been emitted verbatim by
    the Path A passthrough and cannot be recalled.

    This is an accepted, rare edge case.  The test asserts the documented
    behavior so it is explicit, not silent.

    In practice this split is almost never observed because:
      - The Path A → Path B transition fires exactly when the first
        tool_calls delta arrives (different OpenAI SDK field).
      - Leaked opener text and tool_calls deltas are almost never in the
        same chunk.
    """

    def test_a_to_b_boundary_prefix_already_emitted(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Prefix arrives in prose context (A), suffix in TC context (B).

        The prefix '<too' is returned immediately by Path A (passthrough).
        When Path B receives 'l_call>' in TC context, it strips it.
        The net result is '<too' visible to the user — the opener is only
        partially scrubbed.  This is the documented F2 limitation.

        Test asserts this documented (accepted) behavior.
        """
        s.reset()
        # Path A: prose context — '<too' is emitted immediately (passthrough).
        r1 = s.feed("<too", in_toolcall_context=False)
        assert r1 == "<too", (
            f"Path A must pass through '<too' immediately, got {r1!r}"
        )
        # Path B: TC context — 'l_call>' arrives and is stripped.
        # Note: _buf was cleared by the Path A call, so no held context carries over.
        r2 = s.feed("l_call>", in_toolcall_context=True)
        assert r2 == "", (
            f"Path B must strip 'l_call>' in TC ctx, got {r2!r}"
        )
        tail = s.flush()
        # Net result: '<too' was emitted (F2 limitation), 'l_call>' was stripped.
        # The user sees '<too' — partial scrub, which is the documented accepted behavior.
        result = r1 + r2 + tail
        assert result == "<too", (
            f"F2 documented limitation: partial scrub expected '<too', got {result!r}"
        )

    def test_b_to_a_transition_flushes_held_buffer(
        self, s: StreamingToolCallFragmentScrubber
    ) -> None:
        """Buffer held in TC context is flushed when next call is in prose context.

        This is the reverse direction (B→A) handled defensively.
        The held buffer in TC context is flushed and prepended on the
        next prose call — no bytes are dropped.
        """
        s.reset()
        # Path B: TC context — '<too' is held (potential opener prefix).
        r1 = s.feed("<too", in_toolcall_context=True)
        assert r1 == "", "TC ctx holds '<too'"
        # Path A: prose context — held buffer should flush with the new text.
        r2 = s.feed("l text", in_toolcall_context=False)
        # The held '<too' is flushed + new text returned together.
        # '<tool text>' is not an opener-suffix match when reading as prose,
        # so '<too' + 'l text' → '<tool text>' returned verbatim.
        tail = s.flush()
        combined = r1 + r2 + tail
        assert combined == "<tool text", (
            f"B→A transition should flush held buffer, got {combined!r}"
        )
