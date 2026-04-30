"""Tests for defensive terminal control-response stripping in the CLI.

Covers Cursor Position Report (CPR / DSR) responses that occasionally
leak into the input buffer after terminal resize storms or multiplexer
tab switches — see issue #14692.
"""

from cli import _strip_leaked_terminal_responses


class TestStripLeakedTerminalResponses:
    def test_plain_text_unchanged(self):
        text = "hello world"
        assert _strip_leaked_terminal_responses(text) == text

    def test_empty_text(self):
        assert _strip_leaked_terminal_responses("") == ""

    def test_strips_canonical_dsr_response(self):
        # Reports from issue #14692
        text = "\x1b[53;1R"
        assert _strip_leaked_terminal_responses(text) == ""

    def test_strips_dsr_response_in_middle_of_text(self):
        text = "hello\x1b[53;1Rworld"
        assert _strip_leaked_terminal_responses(text) == "helloworld"

    def test_strips_multiple_dsr_responses(self):
        text = "a\x1b[53;1Rb\x1b[51;1Rc\x1b[50;9Rd"
        assert _strip_leaked_terminal_responses(text) == "abcd"

    def test_strips_visible_form_dsr(self):
        # When an upstream filter has already stripped the ESC byte and
        # left the caret-escape representation in place.
        text = "^[[53;1R"
        assert _strip_leaked_terminal_responses(text) == ""

    def test_strips_visible_form_dsr_in_middle_of_text(self):
        text = "typed^[[53;1Rmore"
        assert _strip_leaked_terminal_responses(text) == "typedmore"

    def test_does_not_strip_user_text_with_R(self):
        # Don't over-match; user might genuinely type text containing [N;NR patterns.
        # Our regex requires the leading ESC or caret-escape, so bare
        # "[53;1R" as user text is preserved.
        text = "see section [53;1R for details"
        assert _strip_leaked_terminal_responses(text) == text

    def test_does_not_strip_sgr_sequences(self):
        # Sanity: don't wipe legitimate terminal control sequences that
        # aren't DSR responses.
        text = "\x1b[31mred\x1b[0m"
        assert _strip_leaked_terminal_responses(text) == text

    def test_preserves_multiline_content(self):
        text = "line 1\n\x1b[53;1Rline 2"
        assert _strip_leaked_terminal_responses(text) == "line 1\nline 2"

    def test_strips_sgr_mouse_report_esc_form(self):
        text = "abc\x1b[<65;1;49Mdef"
        assert _strip_leaked_terminal_responses(text) == "abcdef"

    def test_strips_sgr_mouse_report_visible_form(self):
        text = "abc^[[<65;1;49Mdef"
        assert _strip_leaked_terminal_responses(text) == "abcdef"

    def test_strips_sgr_mouse_report_bare_form(self):
        text = "abc<65;1;49Mdef"
        assert _strip_leaked_terminal_responses(text) == "abcdef"

    def test_strips_multiple_concatenated_sgr_mouse_reports(self):
        text = "<65;1;49M<35;1;42Mhello<64;1;40m"
        assert _strip_leaked_terminal_responses(text) == "hello"

    def test_does_not_strip_regular_angle_bracket_text(self):
        text = "render <div class='hero'> literal"
        assert _strip_leaked_terminal_responses(text) == text
