"""Tests for agent.skill_commands.suggest_skill_commands — fuzzy skill
suggestions used by the gateway unknown-command path and the Discord /skill
handler.

Contract: prefix matches outrank substring matches outrank difflib fuzzy
matches; hyphen/underscore input forms are interchangeable; short fragments
never trigger the substring pass; results are capped at *limit* and contain
no duplicates.
"""

from unittest.mock import patch

from agent.skill_commands import suggest_skill_commands

_CATALOG = {
    "/pdf": {},
    "/nano-pdf": {},
    "/brain-pdf": {},
    "/humanizer": {},
    "/docx": {},
    "/xlsx": {},
    "/ocr-and-documents": {},
    "/blond-crm-eintrag": {},
}


def _suggest(command, **kwargs):
    with patch(
        "agent.skill_commands.get_skill_commands", return_value=_CATALOG
    ):
        return suggest_skill_commands(command, **kwargs)


class TestPrefixMatches:
    def test_prefix_match_wins(self):
        # "pd" is a prefix of "pdf" only — must come back first.
        assert _suggest("pd")[0] == "pdf"

    def test_exactish_prefix_prefers_shortest(self):
        # Both "pdf", "nano-pdf" contain pdf, but only "pdf" is a prefix.
        result = _suggest("pdf")
        assert result[0] == "pdf"


class TestSubstringMatches:
    def test_substring_fallback(self):
        # Issue #33822's canonical example: "pdf" should surface nano-pdf.
        result = _suggest("pdf")
        assert "nano-pdf" in result

    def test_substring_requires_three_chars(self):
        # A 2-char fragment must not substring-match half the catalog.
        result = _suggest("cr")
        assert "ocr-and-documents" not in result
        assert "blond-crm-eintrag" not in result

    def test_ocr_example(self):
        assert "ocr-and-documents" in _suggest("ocr")


class TestFuzzyMatches:
    def test_typo_match(self):
        # Transposed typo with no prefix/substring hit.
        assert "humanizer" in _suggest("humanzer")

    def test_garbage_returns_empty(self):
        assert _suggest("zzqqxxyy") == []


class TestNormalization:
    def test_underscores_treated_as_hyphens(self):
        assert "nano-pdf" in _suggest("nano_pdf")

    def test_leading_slash_stripped(self):
        assert "pdf" in _suggest("/pdf")

    def test_empty_input(self):
        assert _suggest("") == []


class TestLimits:
    def test_limit_respected(self):
        assert len(_suggest("pdf", limit=2)) == 2

    def test_no_duplicates(self):
        result = _suggest("pdf")
        assert len(result) == len(set(result))

    def test_empty_catalog(self):
        with patch(
            "agent.skill_commands.get_skill_commands", return_value={}
        ):
            assert suggest_skill_commands("pdf") == []
