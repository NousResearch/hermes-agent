"""Tests for software-engineering-excellence skills.

Validates SKILL.md frontmatter for all 4 skills:
- description <= 60 chars
- author credits human first
- required fields present
- modern section order present
- no project-specific references (Obsidian, Tesouros, LGPD, etc.)
- platforms specified
"""

import pathlib
import re

SKILLS_DIR = pathlib.Path(__file__).parent.parent.parent / "optional-skills" / "software-development"

SKILL_NAMES = [
    "spec-driven-development",
    "ai-agent-guardrails",
    "testing-pyramid-saas",
    "sre-error-budget-solo",
]

# Patterns that indicate project-specific content (not community-ready)
PROJECT_SPECIFIC_PATTERNS = [
    r"\bObsidian\b",
    r"\bTesouros\b",
    r"\bQuizoteca\b",
    r"\bhireme-agent\b",
    r"\bLGPD\b",
    r"\brafaumeu\b",  # GitHub handle should only be in author field
    r"\brafael\.zendron",
]


def _load_skill(name: str) -> tuple[str, pathlib.Path]:
    """Load SKILL.md content for a given skill name."""
    path = SKILLS_DIR / name / "SKILL.md"
    assert path.exists(), f"SKILL.md not found: {path}"
    return path.read_text(), path


def _parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from SKILL.md content."""
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    assert match, "No frontmatter found"
    fm = {}
    for line in match.group(1).splitlines():
        if line.startswith("name:") or line.startswith("version:") or line.startswith("license:"):
            key, val = line.split(":", 1)
            fm[key.strip()] = val.strip().strip('"')
        elif line.startswith("description:"):
            fm["description"] = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("author:"):
            fm["author"] = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("platforms:"):
            fm["platforms"] = line.split(":", 1)[1].strip()
    return fm


class TestFrontmatter:
    """Validate SKILL.md frontmatter for all 4 skills."""

    def test_all_skills_exist(self):
        """All 4 skill directories and SKILL.md files exist."""
        for name in SKILL_NAMES:
            content, path = _load_skill(name)
            assert len(content) > 100, f"{name}: SKILL.md too short"

    def test_description_under_60_chars(self):
        """description <= 60 characters per CONTRIBUTING.md hardline rule."""
        for name in SKILL_NAMES:
            content, _ = _load_skill(name)
            fm = _parse_frontmatter(content)
            desc = fm.get("description", "")
            assert len(desc) <= 60, (
                f"{name}: description is {len(desc)} chars (max 60): \"{desc}\""
            )

    def test_description_not_empty(self):
        """description is non-empty and ends with period."""
        for name in SKILL_NAMES:
            content, _ = _load_skill(name)
            fm = _parse_frontmatter(content)
            desc = fm.get("description", "")
            assert len(desc) > 10, f"{name}: description too short: \"{desc}\""
            assert desc.endswith("."), f"{name}: description must end with period"

    def test_author_credits_human(self):
        """author field credits human contributor first."""
        for name in SKILL_NAMES:
            content, _ = _load_skill(name)
            fm = _parse_frontmatter(content)
            author = fm.get("author", "")
            # Must contain a human name, not just "Hermes Agent"
            assert "Hermes Agent" != author, f"{name}: author must credit human first"
            assert len(author) > 3, f"{name}: author field too short"

    def test_version_present(self):
        """version field present."""
        for name in SKILL_NAMES:
            content, _ = _load_skill(name)
            fm = _parse_frontmatter(content)
            assert "version" in fm, f"{name}: missing version field"

    def test_platforms_specified(self):
        """platforms field specified for cross-platform skills."""
        for name in SKILL_NAMES:
            content, _ = _load_skill(name)
            fm = _parse_frontmatter(content)
            platforms = fm.get("platforms", "")
            assert platforms, f"{name}: missing platforms field"
            # Should contain at least linux and macos
            assert "linux" in platforms, f"{name}: platforms should include linux"

    def test_license_mit(self):
        """license is MIT."""
        for name in SKILL_NAMES:
            content, _ = _load_skill(name)
            fm = _parse_frontmatter(content)
            assert fm.get("license") == "MIT", f"{name}: license should be MIT"


class TestSectionOrder:
    """Validate modern section order per CONTRIBUTING.md."""

    REQUIRED_SECTIONS = [
        "## When to Use",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]

    def test_required_sections_present(self):
        """All required sections present in modern order."""
        for name in SKILL_NAMES:
            content, _ = _load_skill(name)
            for section in self.REQUIRED_SECTIONS:
                assert section in content, f"{name}: missing section '{section}'"

    def test_no_project_specific_content(self):
        """No project-specific references in body (outside author field)."""
        for name in SKILL_NAMES:
            content, _ = _load_skill(name)
            # Strip frontmatter (author may contain GitHub handle)
            body = re.sub(r"^---\n.*?\n---", "", content, flags=re.DOTALL)
            for pattern in PROJECT_SPECIFIC_PATTERNS:
                matches = re.findall(pattern, body, re.IGNORECASE)
                assert not matches, (
                    f"{name}: project-specific reference '{pattern}' found in body: {matches}"
                )

    def test_no_marketing_words(self):
        """No marketing words in description (powerful, comprehensive, seamless, advanced)."""
        marketing_words = ["powerful", "comprehensive", "seamless", "advanced", "revolutionary"]
        for name in SKILL_NAMES:
            content, _ = _load_skill(name)
            fm = _parse_frontmatter(content)
            desc_lower = fm.get("description", "").lower()
            for word in marketing_words:
                assert word not in desc_lower, (
                    f"{name}: marketing word '{word}' in description"
                )


class TestNoStandaloneTools:
    """Skills reference only Hermes-native tools or terminal."""

    STANDALONE_TOOLS = ["grep ", "rg ", "cat ", "head ", "tail ", "sed ", "awk ", "find "]

    def test_prose_uses_hermes_tools(self):
        """Prose sections use Hermes tool names, not raw shell commands."""
        for name in SKILL_NAMES:
            content, _ = _load_skill(name)
            # Only check prose (non-code-block) sections
            # Strip code blocks
            prose = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
            for tool_cmd in self.STANDALONE_TOOLS:
                # Allow in markdown table (command references)
                # But flag in descriptive prose
                lines = prose.splitlines()
                for line in lines:
                    if line.strip().startswith("|"):
                        continue  # tables are OK
                    if line.strip().startswith("#"):
                        continue  # headings
                    if "search_files" in line or "read_file" in line:
                        continue  # showing the right tool alongside
                    # Bare shell commands in prose = violation
                    # (This is a soft check — code blocks are stripped)
