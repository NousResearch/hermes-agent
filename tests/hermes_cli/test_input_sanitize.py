"""Tests for shared user prompt input sanitization."""

import pytest

from hermes_cli.input_sanitize import (
    collapse_repeated_input_artifacts,
    sanitize_user_prompt_text,
    strip_leaked_bracketed_paste_wrappers,
)


# Range corpus from Songkey1 in leve-team/hermes-agent#12; mirrored in Desktop.
ORDINARY_RANGE_TEXTS = (
    "회의는 09:00~18:00 입니다",
    "오전 10:00~11:00, 오후 2:00~3:00",
    "13:00~ 시작",
    "08:01~ 출발",
    "2001~ 2005",
    "10~20개",
    "(100~200)",
    "1.00~1.01",
    "tcp:8000~8010",
    ":00~:30 매시\n:00~ 정각",
    "v1.01~beta",
    "00~06시 심야, 1월 01~ 05일",
    "기간: 2024-01~",
    "값은 [200~300] 사이, [201~ 300]",
    "営業時間は9:00~18:00です",
    "Open 9:00~17:00, pages 100~200",
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
    @pytest.mark.parametrize("text", ORDINARY_RANGE_TEXTS)
    def test_number_ranges_survive_alone_and_inside_canonical_wrappers(self, text):
        assert sanitize_user_prompt_text(text) == text
        for start, end in (("\x1b[200~", "\x1b[201~"), ("^[[200~", "^[[201~")):
            assert sanitize_user_prompt_text(f"{start}{text}{end}") == text

    def test_preserves_every_prefix_of_a_time_or_date_range(self):
        for text in (
            "21:00~21:02",
            "21:01~  21:02",
            "9:00~9:30",
            "12.01~ 12.15",
            "2024-01~2024-02",
        ):
            for end in range(1, len(text) + 1):
                prefix = text[:end]
                assert sanitize_user_prompt_text(prefix) == prefix

    def test_dotted_clock_time_survives_incremental_input(self):
        for text in ("12.01~", "12.01~ 12.15"):
            assert sanitize_user_prompt_text(text) == text

    def test_distinguishes_numeric_text_from_paste_wrappers(self):
        assert sanitize_user_prompt_text("note:00~hello01~") == "note:hello"
        assert (
            sanitize_user_prompt_text("00~21:00~21:02 and 00~x01~")
            == "00~21:00~21:02 and x"
        )
        for text in (
            "id01~21:00~21:02",
            "21:00~21:02[200~tag",
            "00~5 items",
            "count 501~",
        ):
            assert sanitize_user_prompt_text(text) == text

    @pytest.mark.parametrize(
        ("text", "expected"),
        (
            ("\u001b[200~hello\u001b[201~", "hello"),
            ("^[[200~hello^[[201~", "hello"),
            ("prefix [200~hello[201~ suffix", "prefix hello suffix"),
            ("prefix 00~hello world01~ suffix", "prefix hello world suffix"),
            ("^[[200~line 1\nline 2\nline 3^[[201~", "line 1\nline 2\nline 3"),
            ("[200~line 1\nline 2\nline 3[201~", "line 1\nline 2\nline 3"),
        ),
    )
    def test_preserves_cleanup_of_unambiguous_paste_wrappers(self, text, expected):
        assert sanitize_user_prompt_text(text) == expected

    def test_combines_wrapper_strip_and_tail_collapse(self):
        prefix = "hello["
        corrupted = prefix + "~[[e" * 8
        assert sanitize_user_prompt_text(corrupted) == "hello"
