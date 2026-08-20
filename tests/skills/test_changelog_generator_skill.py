"""Tests for optional-skills/software-development/changelog-generator.

Covers two things: the SKILL.md meets the hardline authoring standards
(description length, section order, real related_skills, native-tool
guidance), and scripts/changelog.py merges a new release WITHOUT dropping
existing history. Stdlib + pytest only; no network.
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "optional-skills" / "software-development" / "changelog-generator"
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPTS_DIR = SKILL_DIR / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
import changelog  # noqa: E402


def _frontmatter(text: str) -> str:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "SKILL.md must start with a YAML frontmatter block"
    return m.group(1)


def _field(frontmatter: str, key: str) -> str:
    # Keys may be indented under metadata.hermes (e.g. related_skills).
    m = re.search(rf"^\s*{key}:\s*(.*)$", frontmatter, re.MULTILINE)
    assert m, f"missing frontmatter field: {key}"
    return m.group(1).strip().strip('"').strip("'")


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


# ── Frontmatter / metadata standards ─────────────────────────────────────────

class TestFrontmatter:
    def test_description_within_60_chars(self, skill_text):
        desc = _field(_frontmatter(skill_text), "description")
        assert len(desc) <= 60, f"description is {len(desc)} chars: {desc!r}"

    def test_description_is_one_sentence_ending_in_period(self, skill_text):
        desc = _field(_frontmatter(skill_text), "description")
        assert desc.endswith("."), desc
        # A second sentence shows up as ". " mid-string; a filename dot
        # (".md") does not, so this allows CHANGELOG.md while rejecting
        # multi-sentence descriptions.
        assert ". " not in desc, "description must be a single sentence"

    def test_description_has_no_marketing_words(self, skill_text):
        desc = _field(_frontmatter(skill_text), "description").lower()
        for word in ("powerful", "comprehensive", "seamless", "advanced"):
            assert word not in desc, f"marketing word in description: {word}"

    def test_author_credits_human_first(self, skill_text):
        author = _field(_frontmatter(skill_text), "author")
        assert author.startswith("Burak Koç (@HeLLGURD)"), author
        assert "Hermes Agent" in author

    def test_related_skills_are_real(self, skill_text):
        fm = _frontmatter(skill_text)
        related = _field(fm, "related_skills")
        assert "git-workflow" not in related, "git-workflow is not a real skill"
        assert "github-pr-workflow" in related
        # Referenced skills must resolve in-repo.
        assert (REPO_ROOT / "skills" / "github" / "github-pr-workflow" / "SKILL.md").exists()
        assert (REPO_ROOT / "optional-skills" / "software-development" / "code-wiki" / "SKILL.md").exists()

    def test_platforms_declared(self, skill_text):
        platforms = _field(_frontmatter(skill_text), "platforms")
        for os_name in ("linux", "macos", "windows"):
            assert os_name in platforms


# ── Body / section order ─────────────────────────────────────────────────────

class TestBody:
    REQUIRED = [
        "# Changelog Generator Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]

    def test_required_sections_present(self, skill_text):
        for header in self.REQUIRED:
            assert re.search(rf"^{re.escape(header)}\s*$", skill_text, re.MULTILINE), header

    def test_sections_in_order(self, skill_text):
        positions = [skill_text.index(h) for h in self.REQUIRED]
        assert positions == sorted(positions), "sections are out of order"

    def test_references_native_tools(self, skill_text):
        for tool in ("`read_file`", "`patch`", "`terminal`"):
            assert tool in skill_text, f"missing native tool reference: {tool}"

    def test_no_truncating_read_of_changelog(self, skill_text):
        # The original bug: reading only a few lines then overwriting.
        assert "head -5 CHANGELOG" not in skill_text
        assert "head -5 CHANGELOG.md" not in skill_text


# ── Referenced files exist ───────────────────────────────────────────────────

class TestReferencedFiles:
    def test_script_exists(self):
        assert (SCRIPTS_DIR / "changelog.py").exists()

    def test_reference_exists(self):
        assert (SKILL_DIR / "references" / "categories.md").exists()


# ── Categorization logic ─────────────────────────────────────────────────────

class TestCategorize:
    @pytest.mark.parametrize("subject,expected", [
        ("feat: add dark mode", "Added"),
        ("fix: crash on empty log", "Fixed"),
        ("fix(cli)!: drop --legacy flag", "Breaking Changes"),
        ("refactor: speed up startup", "Changed"),
        ("remove: delete old endpoint", "Removed"),
        ("security: pin starlette", "Security"),
        ("docs: api reference", "Documentation"),
        ("chore: bump deps", "Chores"),
        ("Fixed a nasty bug in the parser", "Fixed"),  # keyword fallback
        ("Introduce batch export", "Added"),           # keyword fallback
        ("Totally unrelated wording", "Other"),
    ])
    def test_categories(self, subject, expected):
        assert changelog.categorize(subject)[0] == expected

    def test_prefix_is_stripped(self):
        _, cleaned = changelog.categorize("feat: add dark mode")
        assert cleaned == "Add dark mode"


class TestRenderSection:
    def test_breaking_changes_first(self):
        commits = [
            {"subject": "feat: add thing"},
            {"subject": "feat!: remove old api"},
        ]
        section = changelog.render_section("v1.2.0", "2026-07-15", commits)
        assert section.index("### Breaking Changes") < section.index("### Added")

    def test_chores_omitted_by_default(self):
        commits = [{"subject": "chore: bump deps"}, {"subject": "feat: real feature"}]
        section = changelog.render_section("v1.0.0", "2026-07-15", commits)
        assert "### Chores" not in section
        section_all = changelog.render_section("v1.0.0", "2026-07-15", commits, include_chores=True)
        assert "### Chores" in section_all

    def test_pr_number_appended(self):
        commits = [{"subject": "fix: patch leak (#312)"}]
        section = changelog.render_section("v1.0.0", "2026-07-15", commits)
        assert "(#312)" in section


# ── History preservation (the core fix) ──────────────────────────────────────

class TestPrependPreservesHistory:
    EXISTING = (
        "# Changelog\n\n"
        "All notable changes to this project will be documented in this file.\n\n"
        "## [Unreleased]\n\n"
        "## [1.0.0] - 2026-01-01\n\n"
        "### Added\n"
        "- Initial release (#1)\n"
    )

    def test_old_entries_survive(self):
        new_section = changelog.render_section(
            "1.1.0", "2026-07-15", [{"subject": "feat: add search (#42)"}]
        )
        result = changelog.prepend_section(self.EXISTING, new_section)
        # Old release and its entry must still be there.
        assert "## [1.0.0] - 2026-01-01" in result
        assert "Initial release (#1)" in result
        # New release present too.
        assert "## [1.1.0] - 2026-07-15" in result
        assert "Add search (#42)" in result

    def test_new_section_above_previous_release(self):
        new_section = changelog.render_section("1.1.0", "2026-07-15", [{"subject": "feat: x"}])
        result = changelog.prepend_section(self.EXISTING, new_section)
        assert result.index("## [1.1.0]") < result.index("## [1.0.0]")

    def test_unreleased_stays_above_new_release(self):
        new_section = changelog.render_section("1.1.0", "2026-07-15", [{"subject": "feat: x"}])
        result = changelog.prepend_section(self.EXISTING, new_section)
        assert result.index("## [Unreleased]") < result.index("## [1.1.0]")

    def test_empty_changelog_gets_header(self):
        new_section = changelog.render_section("1.0.0", "2026-07-15", [{"subject": "feat: first"}])
        result = changelog.prepend_section("", new_section)
        assert result.startswith("# Changelog")
        assert "## [1.0.0]" in result


class TestParseLog:
    def test_parses_pipe_format(self):
        line = "abc123|feat: add x|Jane|jane@example.com|2026-07-15"
        commits = changelog.parse_log(line)
        assert len(commits) == 1
        assert commits[0]["hash"] == "abc123"
        assert commits[0]["subject"] == "feat: add x"

    def test_skips_blank_lines(self):
        assert changelog.parse_log("\n\n") == []
