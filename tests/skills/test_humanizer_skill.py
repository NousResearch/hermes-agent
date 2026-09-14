"""Structural tests for the bundled humanizer skill.

The skill is a Hermes port of blader/humanizer (MIT). These guards keep the
sync honest: the upstream pattern set must stay complete, and the Hermes
adaptations (native tool references, the marketing-cliche list, attribution)
must survive future upstream syncs.
"""

from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "creative" / "humanizer"
SKILL_MD = SKILL_DIR / "SKILL.md"


@pytest.fixture(scope="module")
def text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter(text: str) -> str:
    return text.split("---", 2)[1]


class TestFrontmatter:
    def test_required_fields(self, text):
        fm = _frontmatter(text)
        for field in ("name:", "description:", "version:", "author:", "license:", "platforms:"):
            assert field in fm, f"missing {field}"

    def test_tracks_upstream_version(self, text):
        assert "version: 3.0.0" in _frontmatter(text)

    def test_credits_original_author(self, text):
        fm = _frontmatter(text)
        assert "Siqi Chen" in fm and "@blader" in fm

    def test_hermes_metadata(self, text):
        fm = _frontmatter(text)
        assert "category: creative" in fm
        assert "homepage: https://github.com/blader/humanizer" in fm


class TestPatternCoverage:
    """The 3.0.0 upstream layout: 25 numbered patterns in five families."""

    FAMILIES = [
        "## A. Staging instead of stating",
        "## B. Rhythm by rule",
        "## C. Inflation and borrowed authority",
        "## D. Formatting by rule",
        "## E. Leftovers from the chat and the draft",
    ]

    PATTERNS = {
        1: "Not X but Y",
        2: "One-line closers and dramatic fragments",
        3: "Sayings that sound deep",
        4: "Staged run-up before the point",
        5: "Arguing with no one",
        6: "Forced triads",
        7: "Repeated sentence openings",
        8: "Dashes as the universal connector",
        9: "Stacked qualifiers",
        10: "Hyphenated pairs everywhere",
        11: "Passive voice and missing subjects",
        12: "Overused AI words",
        13: "Inflated significance",
        14: "Vague connection or association",
        15: "Shallow -ing riders",
        16: "Sales language",
        17: "Borrowed authority",
        18: "Avoiding is, are, and has",
        19: "Bold as decoration",
        20: "Decorative headings",
        21: "Curly quotation marks",
        22: "Chatbot residue",
        23: "Knowledge-limit disclaimers and guesses",
        24: "A heading repeated in the first sentence",
        25: "Writing about the previous version",
    }

    def test_all_families_present(self, text):
        for fam in self.FAMILIES:
            assert fam in text, f"missing family heading: {fam}"

    @pytest.mark.parametrize("num,title", sorted(PATTERNS.items()))
    def test_pattern_present(self, text, num, title):
        assert f"### {num}. {title}" in text

    def test_worked_sections_exist(self, text):
        for section in ("## How to work", "### Voice", "### What to return",
                        "## When not to act", "## Attribution"):
            assert section in text


class TestHermesAdaptations:
    """Local additions that a naive upstream sync would silently drop."""

    def test_file_mode_names_native_tools(self, text):
        file_mode = text.split("**File mode.**", 1)[1].split("\n\n", 1)[0]
        for tool in ("read_file", "patch", "write_file"):
            assert f"`{tool}`" in file_mode

    def test_marketing_cliche_list_kept(self, text):
        assert "Marketing and blog cliches" in text
        for cliche in ("game-changer", "circle back", "lean into"):
            assert cliche in text

    def test_trigger_list_kept(self, text):
        assert "un-ChatGPT" in text

    def test_curly_quote_pattern_example_differs(self, text):
        """Regression for the pattern-19 bug (upstream #219, Hermes PR #72921):
        Before/After must not be byte-identical or the example shows nothing."""
        sec = text.split("### 21. Curly quotation marks", 1)[1].split("###", 1)[0]
        before = sec.split("**Before:**", 1)[1].split("**After:**", 1)[0]
        after = sec.split("**After:**", 1)[1]
        assert "\u201c" in before, "Before example lost its curly quotes"
        assert '"' in after and "\u201c" not in after

    def test_attribution_records_upstream_sync(self, text):
        assert "blader/humanizer" in text
        assert "version 3.0.0" in text

    def test_license_file_ships(self):
        assert (SKILL_DIR / "LICENSE").exists()

    def test_no_machine_local_paths(self, text):
        for needle in ("/home/", "/Users/"):
            assert needle not in text
