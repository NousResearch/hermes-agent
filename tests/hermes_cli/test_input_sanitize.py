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

    def test_strips_a_repeated_degraded_opening_wrapper(self):
        """A leak that emits the marker twice left one behind.

        The boundary character is consumed and re-emitted by the substitution, so on the
        second marker the preceding character is the first marker's "~" rather than a
        boundary, and it never matched again.
        """
        assert strip_leaked_bracketed_paste_wrappers("[200~[200~hello") == "hello"
        assert strip_leaked_bracketed_paste_wrappers("[200~[200~[200~hello") == "hello"
        assert strip_leaked_bracketed_paste_wrappers("prefix [200~[200~hello") == "prefix hello"

    def test_strips_repeated_degraded_wrapper_fragments(self):
        assert strip_leaked_bracketed_paste_wrappers("00~00~hello") == "hello"
        assert strip_leaked_bracketed_paste_wrappers("hello01~01~") == "hello"
        assert (
            strip_leaked_bracketed_paste_wrappers("00~00~00~hello world01~01~01~")
            == "hello world"
        )

    def test_strips_a_repeated_wrapper_pair(self):
        assert strip_leaked_bracketed_paste_wrappers("[200~[200~hello[201~[201~") == "hello"

    def test_repeats_do_not_widen_the_boundary_rule(self):
        """Repetition must not make an embedded literal strippable."""
        text = "literal[200~[200~tag should stay"
        assert strip_leaked_bracketed_paste_wrappers(text) == text


class TestCollapseRepeatedInputArtifacts:
    def test_issue_62557_corruption_tail(self):
        prefix = "需要时随时叫我。"
        tail = "[e~[[e" + "~[[e" * 20
        assert collapse_repeated_input_artifacts(prefix + tail) == prefix


    def test_trailing_punctuation_preserved(self):
        assert collapse_repeated_input_artifacts("wait....") == "wait...."


class TestSanitizeUserPromptText:
    def test_combines_wrapper_strip_and_tail_collapse(self):
        prefix = "hello["
        corrupted = prefix + "~[[e" * 8
        assert sanitize_user_prompt_text(corrupted) == "hello"
