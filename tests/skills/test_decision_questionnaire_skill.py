"""Tests for optional-skills/productivity/decision-questionnaire."""

import re
from pathlib import Path

import pytest
import yaml

SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "productivity"
    / "decision-questionnaire"
    / "SKILL.md"
)


def _frontmatter():
    text = SKILL_MD.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


class TestFrontmatter:
    def test_name_matches_directory(self):
        assert _frontmatter()["name"] == "decision-questionnaire"

    def test_description_length_and_period(self):
        desc = _frontmatter()["description"]
        assert len(desc) <= 60
        assert desc.endswith(".")

    def test_license_and_platforms(self):
        fm = _frontmatter()
        assert fm["license"] == "MIT"
        assert set(fm["platforms"]) == {"linux", "macos", "windows"}


class TestBody:
    def _body(self):
        return SKILL_MD.read_text(encoding="utf-8")

    def test_template_sections_present(self):
        body = self._body()
        for section in ("## Context", "## How to answer", "## Anything else?"):
            assert section in body, f"template missing {section}"

    def test_output_filename_convention(self):
        assert "decision-questionnaire-<slug>.md" in self._body()

    def test_interview_the_send_principle(self):
        assert "Interview the Send" in self._body()

    def test_no_upstream_harness_residue(self):
        lower = self._body().lower()
        for token in ("claude", "slash command", "disable-model-invocation"):
            assert token not in lower, f"upstream residue: {token}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
