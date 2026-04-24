"""Tests for extract_skill_description truncation (#13944)."""
import pytest
from agent.skill_utils import extract_skill_description

class TestExtractSkillDescriptionTruncation:
    def test_short_description_unchanged(self):
        assert extract_skill_description({"description": "Short"}) == "Short"

    def test_70_char_description_preserved(self):
        desc = "A" * 70
        assert extract_skill_description({"description": desc}) == desc

    def test_120_char_preserved(self):
        desc = "A" * 120
        assert extract_skill_description({"description": desc}) == desc

    def test_121_char_truncated(self):
        desc = "A" * 121
        result = extract_skill_description({"description": desc})
        assert result == "A" * 117 + "..."
        assert len(result) == 120

    def test_empty(self):
        assert extract_skill_description({}) == ""
        assert extract_skill_description({"description": ""}) == ""
