"""Tests for shared user prompt input sanitization."""

from hermes_cli.input_sanitize import (
    collapse_repeated_input_artifacts,
    decode_csi_u_in_paste,
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


class TestDecodeCsiUInPaste:
    def test_plain_text_unchanged(self):
        assert decode_csi_u_in_paste("line one\nline two") == "line one\nline two"

    def test_ctrl_j_and_ctrl_m_become_newlines(self):
        # tmux `extended-keys always` re-encodes a pasted newline as Ctrl+J
        assert decode_csi_u_in_paste("a\x1b[106;5ub\x1b[109;5uc") == "a\nb\nc"

    def test_ctrl_i_becomes_tab(self):
        assert decode_csi_u_in_paste("col1\x1b[105;5ucol2") == "col1\tcol2"

    def test_lock_state_modifiers_still_decode(self):
        # kitty ORs CapsLock (+64) / NumLock (+128) into the modifier
        assert decode_csi_u_in_paste("x\x1b[106;69uy\x1b[106;197uz") == "x\ny\nz"

    def test_other_key_encodings_are_dropped(self):
        assert decode_csi_u_in_paste("p\x1b[97;3uq") == "pq"

    def test_literal_text_without_escape_is_untouched(self):
        text = "a literal [106;5u in prose"
        assert decode_csi_u_in_paste(text) == text
