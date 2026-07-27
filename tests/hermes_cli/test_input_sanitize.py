"""Tests for shared user prompt input sanitization."""

from hermes_cli.input_sanitize import (
    collapse_repeated_input_artifacts,
    sanitize_user_prompt_text,
    strip_leaked_bracketed_paste_wrappers,
)


class TestStripLeakedBracketedPasteWrappers:
    def test_plain_text_unchanged(self):
        assert strip_leaked_bracketed_paste_wrappers("hello world") == "hello world"

    def test_strips_canonical_escape_wrappers(self):
        assert strip_leaked_bracketed_paste_wrappers("\x1b[200~hello\x1b[201~") == "hello"

    def test_strips_visible_caret_escape_wrappers(self):
        assert strip_leaked_bracketed_paste_wrappers("^[[200~hello^[[201~") == "hello"

    def test_does_not_strip_non_wrapper_bracket_forms_in_normal_text(self):
        text = "literal[200~tag and literal[201~tag should stay"
        assert strip_leaked_bracketed_paste_wrappers(text) == text


class TestCollapseRepeatedInputArtifacts:
    def test_issue_62557_corruption_tail(self):
        prefix = "需要时随时叫我。"
        tail = "[e~[[e" + "~[[e" * 20
        assert collapse_repeated_input_artifacts(prefix + tail) == prefix

    def test_plain_text_unchanged(self):
        text = "build00~tag should stay"
        assert collapse_repeated_input_artifacts(text) == text

    def test_mid_string_marker_followed_by_suffix_preserved(self):
        text = "notes ~[[e more text here"
        assert collapse_repeated_input_artifacts(text) == text

    def test_trailing_punctuation_preserved(self):
        assert collapse_repeated_input_artifacts("wait....") == "wait...."

    def test_insufficient_tail_repeats_preserved(self):
        text = "hello~[[e~[[e"
        assert collapse_repeated_input_artifacts(text) == text


class TestSanitizeUserPromptText:
    def test_combines_wrapper_strip_and_tail_collapse(self):
        prefix = "hello["
        corrupted = prefix + "~[[e" * 8
        assert sanitize_user_prompt_text(corrupted) == "hello"
