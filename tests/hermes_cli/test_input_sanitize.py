"""Tests for shared user prompt input sanitization."""

from hermes_cli.input_sanitize import (
    collapse_repeated_input_artifacts,
    sanitize_user_prompt_text,
    strip_leaked_bracketed_paste_wrappers,
)


class TestStripLeakedBracketedPasteWrappers:
    def test_plain_text_unchanged(self):
        assert strip_leaked_bracketed_paste_wrappers("hello world") == "hello world"



    def test_does_not_strip_non_wrapper_bracket_forms_in_normal_text(self):
        text = "literal[200~tag and literal[201~tag should stay"
        assert strip_leaked_bracketed_paste_wrappers(text) == text


class TestCollapseRepeatedInputArtifacts:
    def test_issue_62557_corruption_tail(self):
        prefix = "需要时随时叫我。"
        tail = "[e~[[e" + "~[[e" * 20
        assert collapse_repeated_input_artifacts(prefix + tail) == prefix


    def test_trailing_punctuation_preserved(self):
        assert collapse_repeated_input_artifacts("wait....") == "wait...."


class TestSanitizeUserPromptText:
    def test_preserves_every_prefix_of_a_clock_time_range(self):
        for text in ("21:00~21:02", "21:01~  21:02", "9:00~9:30"):
            for end in range(1, len(text) + 1):
                prefix = text[:end]
                assert sanitize_user_prompt_text(prefix) == prefix

    def test_distinguishes_clock_text_from_paste_wrappers(self):
        assert sanitize_user_prompt_text("note:00~hello01~") == "note:hello"
        assert sanitize_user_prompt_text("00~21:00~21:02 and 00~x01~") == "21:00~21:02 and x"
        for text in ("id01~21:00~21:02", "21:00~21:02[200~tag"):
            assert sanitize_user_prompt_text(text) == text

    def test_combines_wrapper_strip_and_tail_collapse(self):
        prefix = "hello["
        corrupted = prefix + "~[[e" * 8
        assert sanitize_user_prompt_text(corrupted) == "hello"
