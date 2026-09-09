"""Tests for defensive terminal control-response stripping in the CLI.

Covers Cursor Position Report (CPR / DSR) responses that occasionally
leak into the input buffer after terminal resize storms or multiplexer
tab switches — see issue #14692.
"""

from cli import _strip_leaked_terminal_responses_with_meta


def _strip_leaked_terminal_responses(text: str) -> str:
    cleaned, _ = _strip_leaked_terminal_responses_with_meta(text)
    return cleaned


class TestStripLeakedTerminalResponses:


    def test_strips_canonical_dsr_response(self):
        # Reports from issue #14692
        text = "\x1b[53;1R"
        assert _strip_leaked_terminal_responses(text) == ""


    def test_strips_multiple_dsr_responses(self):
        text = "a\x1b[53;1Rb\x1b[51;1Rc\x1b[50;9Rd"
        assert _strip_leaked_terminal_responses(text) == "abcd"






    def test_strips_sgr_mouse_report_esc_form(self):
        text = "abc\x1b[<65;1;49Mdef"
        assert _strip_leaked_terminal_responses(text) == "abcdef"



    def test_strips_sgr_mouse_report_with_large_coordinates(self):
        text = "abc\x1b[<10000;12345;98765Mdef"
        assert _strip_leaked_terminal_responses(text) == "abcdef"

    def test_strips_multiple_concatenated_sgr_mouse_reports(self):
        text = "<65;1;49M<35;1;42Mhello<64;1;40m"
        assert _strip_leaked_terminal_responses(text) == "hello"

