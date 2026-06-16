"""Tests for defensive bracketed-paste wrapper stripping in the CLI."""

from cli import _strip_leaked_bracketed_paste_wrappers


class TestStripLeakedBracketedPasteWrappers:
    def test_plain_text_unchanged(self):
        text = "hello world"
        assert _strip_leaked_bracketed_paste_wrappers(text) == text

    def test_strips_canonical_escape_wrappers(self):
        text = "\x1b[200~hello\x1b[201~"
        assert _strip_leaked_bracketed_paste_wrappers(text) == "hello"

    def test_strips_visible_caret_escape_wrappers(self):
        text = "^[[200~hello^[[201~"
        assert _strip_leaked_bracketed_paste_wrappers(text) == "hello"

    def test_strips_degraded_bracket_only_wrappers(self):
        text = "[200~hello[201~"
        assert _strip_leaked_bracketed_paste_wrappers(text) == "hello"

    def test_strips_degraded_bracket_only_wrappers_after_whitespace(self):
        text = "prefix [200~hello[201~ suffix"
        assert _strip_leaked_bracketed_paste_wrappers(text) == "prefix hello suffix"

    def test_strips_wrapper_fragments_at_boundaries(self):
        text = "00~hello world01~"
        assert _strip_leaked_bracketed_paste_wrappers(text) == "hello world"

    def test_strips_wrapper_fragments_after_whitespace(self):
        text = "prefix 00~hello world01~ suffix"
        assert _strip_leaked_bracketed_paste_wrappers(text) == "prefix hello world suffix"

    def test_does_not_strip_non_wrapper_00_tilde_in_normal_text(self):
        text = "build00~tag should stay"
        assert _strip_leaked_bracketed_paste_wrappers(text) == text

    def test_does_not_strip_non_wrapper_bracket_forms_in_normal_text(self):
        text = "literal[200~tag and literal[201~tag should stay"
        assert _strip_leaked_bracketed_paste_wrappers(text) == text

    def test_preserves_multiline_content_while_stripping_wrappers(self):
        text = "^[[200~line 1\nline 2\nline 3^[[201~"
        assert _strip_leaked_bracketed_paste_wrappers(text) == "line 1\nline 2\nline 3"

    def test_preserves_multiline_content_while_stripping_degraded_bracket_only_wrappers(self):
        text = "[200~line 1\nline 2\nline 3[201~"
        assert _strip_leaked_bracketed_paste_wrappers(text) == "line 1\nline 2\nline 3"
