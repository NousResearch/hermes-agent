"""Tests for shared user prompt input sanitization."""

from hermes_cli.input_sanitize import (
    replace_lone_surrogates,
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
    def test_combines_wrapper_strip_and_tail_collapse(self):
        prefix = "hello["
        corrupted = prefix + "~[[e" * 8
        assert sanitize_user_prompt_text(corrupted) == "hello"


class TestReplaceLoneSurrogates:
    def test_lone_surrogate_becomes_replacement_char(self):
        assert replace_lone_surrogates("a\udcffb") == "a\ufffdb"

    def test_clean_text_is_returned_unchanged(self):
        assert replace_lone_surrogates("git status") == "git status"

    def test_is_idempotent(self):
        once = replace_lone_surrogates("a\udcffb")
        assert replace_lone_surrogates(once) == once

    def test_result_is_utf8_encodable(self):
        replace_lone_surrogates("a\udcffb").encode("utf-8")  # must not raise

    def test_valid_surrogate_pair_becomes_the_real_character(self):
        # Two code points forming a valid pair are one non-BMP character (U+1F600).
        # Flattening them to U+FFFD would corrupt text that was never broken.
        pair = "\ud83d\ude00"
        repaired = replace_lone_surrogates(pair)
        assert repaired == "\U0001F600"
        assert repaired.encode("utf-8") == b"\xf0\x9f\x98\x80"

    def test_lone_surrogate_next_to_a_pair(self):
        assert replace_lone_surrogates("\ud83d\ude00\udcff") == "\U0001F600\ufffd"

    def test_prompt_sanitizer_drops_lone_surrogates(self):
        assert sanitize_user_prompt_text("a\udcffb") == "a\ufffdb"

    def test_prompt_sanitizer_repairs_a_valid_pair(self):
        assert sanitize_user_prompt_text("\ud83d\ude00") == "\U0001F600"
