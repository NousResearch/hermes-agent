"""Tests for skills/software-development/adversarial-self-review/SKILL.md

Validates SKILL.md metadata, structure, and HARDLINE rule compliance
for a pure prompt-pattern skill (no scripts, no external dependencies).
"""

import re
from pathlib import Path

import pytest


def _load_yaml():
    """Load PyYAML lazily — it's a core Hermes dependency but import only when needed."""
    import yaml as _yaml
    return _yaml


SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "adversarial-self-review"
)
SKILL_MD = SKILL_DIR / "SKILL.md"


def _read_skill():
    """Read SKILL.md and return (frontmatter_dict, body_text)."""
    yaml = _load_yaml()
    text = SKILL_MD.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("SKILL.md missing YAML frontmatter delimiters")
    frontmatter = yaml.safe_load(parts[1])
    body = parts[2]
    return frontmatter, body


class TestSkillExists:
    def test_skill_md_present(self):
        assert SKILL_MD.exists(), f"SKILL.md not found at {SKILL_MD}"
        assert SKILL_MD.is_file()

    def test_no_scripts_dir(self):
        """Prompt-pattern skill — should have no scripts/ directory."""
        scripts = SKILL_DIR / "scripts"
        assert not scripts.exists(), (
            "This is a pure prompt-pattern skill. "
            "Remove scripts/ — if scripts are needed, add them with tests."
        )


class TestFrontmatter:
    @pytest.fixture(autouse=True)
    def load(self):
        self.fm, self.body = _read_skill()

    def test_name_matches_directory(self):
        assert self.fm["name"] == "adversarial-self-review"

    def test_description_length(self):
        desc = self.fm["description"]
        assert len(desc) <= 60, (
            f"description is {len(desc)} chars (max 60): {desc!r}"
        )

    def test_description_ends_with_period(self):
        assert self.fm["description"].endswith(".")

    def test_description_is_one_sentence(self):
        desc = self.fm["description"]
        inner = desc.rstrip(".")
        assert "." not in inner, (
            f"description should be one sentence: {desc!r}"
        )

    def test_description_no_marketing_words(self):
        desc = self.fm["description"].lower()
        banned = ["powerful", "comprehensive", "seamless", "advanced"]
        for word in banned:
            assert word not in desc, (
                f"description has marketing word '{word}': {self.fm['description']!r}"
            )

    def test_version_present(self):
        assert "version" in self.fm

    def test_author_human_first(self):
        author = self.fm.get("author", "")
        assert author != "Hermes Agent", (
            "author must credit human first, not 'Hermes Agent'"
        )
        assert len(author) > 0

    def test_license_is_mit(self):
        assert self.fm.get("license", "").upper() == "MIT"

    def test_tags_present(self):
        tags = self.fm.get("metadata", {}).get("hermes", {}).get("tags", [])
        assert len(tags) >= 2, "skill should have at least 2 tags"
        assert "Software Development" in tags

    def test_related_skills_valid(self):
        related = (
            self.fm.get("metadata", {})
            .get("hermes", {})
            .get("related_skills", [])
        )
        assert "requesting-code-review" in related, (
            "should reference requesting-code-review as related skill"
        )


class TestRequiredSections:
    """HARDLINE rule 5: modern section order."""

    REQUIRED = [
        "When to Use",
        "Prerequisites",
        "How to Run",
        "Quick Reference",
        "Procedure",
        "Pitfalls",
        "Verification",
    ]

    @pytest.fixture(autouse=True)
    def load(self):
        self.fm, self.body = _read_skill()

    @pytest.mark.parametrize("section", REQUIRED)
    def test_section_present(self, section):
        assert f"## {section}" in self.body, (
            f"Missing required section: ## {section}"
        )

    def test_section_order(self):
        """Sections must appear in the order specified by HARDLINE rule 5."""
        positions = {}
        for section in self.REQUIRED:
            idx = self.body.find(f"## {section}")
            if idx >= 0:
                positions[section] = idx
        ordered = sorted(positions.items(), key=lambda x: x[1])
        ordered_names = [name for name, _ in ordered]
        present_required = [s for s in self.REQUIRED if s in positions]
        assert ordered_names == present_required, (
            f"Sections out of order. Expected: {present_required}, "
            f"Got: {ordered_names}"
        )


class TestToolReferences:
    """HARDLINE rule 2: only native Hermes tools, no shell commands.

    Only checks backtick-wrapped commands (`` `cmd` ``) — the HARDLINE
    violation is specifically naming a shell utility where a Hermes tool
    should be named. Plain English words like "find every bug" are NOT
    violations; `` `find` `` used as a tool instruction IS.
    """

    # Mapping from shell command (when used as a tool instruction) → native tool
    SHELL_TO_NATIVE = {
        "grep": "search_files",
        "rg": "search_files",
        "cat": "read_file",
        "head": "read_file",
        "tail": "read_file",
        "sed": "patch",
        "awk": "patch",
        "find": "search_files (target='files')",
        "ls": "search_files (target='files')",
    }

    @pytest.fixture(autouse=True)
    def load(self):
        self.fm, self.body = _read_skill()

    def test_native_tools_referenced(self):
        """Must reference at least one native Hermes tool."""
        native = ["delegate_task", "write_file", "patch", "read_file",
                   "terminal", "web_extract", "web_search", "search_files"]
        body_lower = self.body.lower()
        found = [t for t in native if t.lower() in body_lower]
        assert len(found) > 0, (
            "SKILL.md must reference at least one native Hermes tool"
        )

    def test_no_shell_commands_in_backticks(self):
        """Must not instruct agent to use `grep` instead of `search_files`, etc.

        Only checks backtick-wrapped references — plain English uses of
        these words (e.g. "find every bug") are not violations.
        """
        # Extract all backtick-wrapped terms
        backtick_terms = set(re.findall(r'`([^`]+)`', self.body))
        violations = []
        for term in backtick_terms:
            term_lower = term.lower().strip()
            if term_lower in self.SHELL_TO_NATIVE:
                native = self.SHELL_TO_NATIVE[term_lower]
                violations.append(f"`{term}` → use `{native}` instead")
        assert not violations, (
            f"Backtick-wrapped shell commands found — use native Hermes tools:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )


class TestBodyQuality:
    @pytest.fixture(autouse=True)
    def load(self):
        self.fm, self.body = _read_skill()

    def test_line_count(self):
        """Target ~100-200 lines for a complex skill (HARDLINE rule 5)."""
        lines = self.body.strip().split("\n")
        assert len(lines) <= 250, (
            f"Body is {len(lines)} lines — too long for a prompt-pattern skill"
        )

    def test_first_heading_is_title(self):
        """First heading must be '# Adversarial Self-Review Skill'."""
        for line in self.body.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# "):
                assert "Adversarial Self-Review" in stripped, (
                    f"First heading should be skill title, got: {stripped}"
                )
                break

    def test_no_marketing_prose(self):
        """Intro should state capability, not use marketing language."""
        intro_paragraphs = self.body.split("\n\n")[:3]
        banned = ["powerful", "comprehensive", "seamless", "advanced",
                   "revolutionary", "game-changing"]
        for para in intro_paragraphs:
            para_lower = para.lower()
            for word in banned:
                assert word not in para_lower, (
                    f"Marketing word '{word}' in: {para[:80]}..."
                )

    def test_delegate_task_referenced_in_procedure(self):
        """Procedure section must reference delegate_task."""
        procedure_start = self.body.find("## Procedure")
        procedure_text = self.body[procedure_start:]
        assert "delegate_task" in procedure_text, (
            "Procedure section must reference delegate_task"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
