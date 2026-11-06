"""Structural tests for the bundled humanizer skill.

The skill is a Hermes port of blader/humanizer (MIT). These guards pin the
relationships that actually break during an upstream sync: that pattern
citations resolve, the pattern bank is contiguous, and every pattern carries a
problem statement and an example. They deliberately avoid pinning exact
renumberings or wording.
"""

from pathlib import Path
import re

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "creative" / "humanizer"
SKILL_MD = SKILL_DIR / "SKILL.md"



import pytest

@pytest.fixture(scope="module")
def text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter(text: str) -> str:
    return text.split("---", 2)[1]


def _pattern_numbers(text: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"^### (\d+)\. ", text, re.MULTILINE)]


def _pattern_sections(text: str) -> dict[int, str]:
    # Split body into per-pattern chunks keyed by number.
    matches = list(re.finditer(r"^### (\d+)\. .*$", text, re.MULTILINE))
    out = {}
    for i, m in enumerate(matches):
        n = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[n] = text[start:end]
    return out


class TestFrontmatter:
    def test_required_fields(self, text):
        fm = _frontmatter(text)
        for field in ("name:", "description:", "version:", "author:", "license:", "platforms:"):
            assert field in fm, f"missing {field}"

    def test_tracks_upstream_version(self, text):
        fm = _frontmatter(text)
        assert "version: 3.1.0" in fm

    def test_credits_original_author(self, text):
        fm = _frontmatter(text)
        assert "Siqi Chen" in fm and "@blader" in fm

    def test_hermes_metadata(self, text):
        fm = _frontmatter(text)
        assert "category: creative" in fm
        assert "homepage: https://github.com/blader/humanizer" in fm


class TestPatternBank:
    def test_families_present(self, text):
        for fam in ("## A. ", "## B. ", "## C. ", "## D. ", "## E. ", "## F. "):
            assert fam in text, f"missing family {fam}"

    def test_contiguous_numbering(self, text):
        nums = _pattern_numbers(text)
        assert nums == list(range(1, len(nums) + 1)), "pattern numbers must be 1..N with no gaps"

    def test_nonempty_bank(self, text):
        assert len(_pattern_numbers(text)) >= 25

    def test_each_pattern_has_problem_statement(self, text):
        for n, chunk in _pattern_sections(text).items():
            assert "**Problem:**" in chunk, f"pattern {n} missing problem statement"

    def test_each_pattern_has_example(self, text):
        for n, chunk in _pattern_sections(text).items():
            assert "Before" in chunk and "After:" in chunk, f"pattern {n} missing Before/After example"

    def test_citations_resolve(self, text):
        nums = set(_pattern_numbers(text))
        for m in re.finditer(r"§(\d+)", text):
            n = int(m.group(1))
            assert n in nums, f"citation §{n} does not resolve to a pattern heading"


class TestHermesAdaptations:
    def test_native_tool_references(self, text):
        for tool in ("read_file", "patch", "write_file"):
            assert tool in text

    def test_trigger_list(self, text):
        assert "humanize" in text.lower()
        assert "de-AI" in text or "de-slop" in text

    def test_marketing_cliche_list(self, text):
        assert "Marketing and blog cliches" in text

    def test_voice_and_soul_section(self, text):
        assert "Personality and soul" in text

    def test_when_not_to_use_section(self, text):
        assert "## When NOT to Use" in text

    def test_attribution_mentions_extras(self, text):
        assert "Hermes-native" in text or "Hermes Agent" in text
        assert "3.1.0" in text


class TestCurlyQuotesRegression:
    def test_pattern_21_curly_quotes_example(self, text):
        sections = _pattern_sections(text)
        # pattern 21 is the curly-quotes pattern in 3.1.0
        assert 21 in sections
        chunk = sections[21]
        # Before example should contain curly quotes
        assert "\u201c" in chunk or '"' in chunk
        assert "He said" in chunk
