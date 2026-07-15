"""Tests for the skill-auditor script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

# ── Load the module under test ──────────────────────────────────────────────
# The module lives under skills/software-development/skill-auditor/ which
# contains hyphens and isn't a valid Python package path, so we load via
# importlib from the file location.
_AUDITOR_PATH = (
    Path(__file__).parent.parent.parent
    / "skills"
    / "software-development"
    / "skill-auditor"
    / "scripts"
    / "audit_skills.py"
)

_spec = importlib.util.spec_from_file_location("audit_skills", _AUDITOR_PATH)
# Register module in sys.modules BEFORE exec_module so dataclass decorators
# (which introspect cls.__module__) don't hit a NoneType error.
audit_skills = importlib.util.module_from_spec(_spec)
sys.modules["audit_skills"] = audit_skills
_spec.loader.exec_module(audit_skills)  # type: ignore[union-attr]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_skill(content: str, **fields: Any) -> str:
    """Build a minimal SKILL.md string with given body and frontmatter overrides."""
    name = fields.get("name", "test-skill")
    desc = fields.get("description", "A test skill for auditing.")
    version = fields.get("version", "1.0.0")
    author = fields.get("author", "Test Author")
    fm = (
        "---\n"
        f"name: {name}\n"
        f'description: "{desc}"\n'
        f"version: {version}\n"
        f"author: {author}\n"
        "---\n\n"
    )
    return fm + content


_BODY_OK = """\
## When to Use

- Some use case

## Prerequisites

- Python 3.11+

## How to Run

```bash
python3 audit_skills.py
```

## Quick Reference

| Metric | What it checks |
|--------|---------------|
| **size** | Line count |

## Procedure

1. Run the audit

## Pitfalls

- Watch out for common pitfalls here.

## Verification

Run the script and check output.
"""


# ── Tests: regex fix (PR #49795) ───────────────────────────────────────────


class TestToolReferenceRegex:
    """Verify the regex fix: prose refs are caught, backticked ones are not."""

    def test_prose_grep_is_detected(self):
        """Plain prose 'grep' in text should be flagged."""
        issues = audit_skills.check_tool_references("you can use grep to search files")
        assert any("grep" in i.message for i in issues)
        assert any("search_files" in i.message for i in issues)

    def test_backtick_grep_is_not_detected(self):
        """Backticked `grep` should NOT be flagged."""
        issues = audit_skills.check_tool_references("use `grep` inside backticks")
        assert not any("grep" in i.message for i in issues)

    def test_all_banned_tools_detected_in_prose(self):
        """Every banned tool should be caught when used as plain prose."""
        for banned, replacement in audit_skills.BANNED_TOOL_REFS.items():
            text = f"you can run {banned} to find things"
            issues = audit_skills.check_tool_references(text)
            assert any(banned in i.message for i in issues), (
                f"'{banned}' should be detected in prose"
            )
            assert any(replacement in i.message for i in issues), (
                f"'{banned}' should suggest '{replacement}'"
            )

    def test_banned_tools_not_detected_in_backticks(self):
        """Banned tool names inside backticks should be ignored."""
        for banned in audit_skills.BANNED_TOOL_REFS:
            text = f"use `{banned}` in backtick notation"
            issues = audit_skills.check_tool_references(text)
            assert not any(banned in i.message for i in issues), (
                f"'{banned}' inside backticks should not be flagged"
            )

    def test_banned_tools_in_code_blocks_safe(self):
        """Banned tool names inside ``` code blocks should be safe."""
        text = """\
Some prose here.

```bash
grep -r pattern .
cat file.txt
sed -i 's/foo/bar/' file
```

More prose.
"""
        issues = audit_skills.check_tool_references(text)
        assert not any(
            i.severity == "warning" for i in issues
        ), "Should not flag tools inside code blocks"


# ── Tests: frontmatter checks ──────────────────────────────────────────────


class TestFrontmatter:
    def test_valid_frontmatter_no_issues(self):
        """A complete frontmatter block should produce no issues."""
        content = _make_skill(_BODY_OK)
        issues = audit_skills.check_frontmatter(content)
        assert len(issues) == 0, f"Got unexpected issues: {issues}"

    def test_missing_frontmatter(self):
        """Content without leading --- should produce an error."""
        issues = audit_skills.check_frontmatter("no frontmatter here")
        assert any(i.severity == "error" for i in issues)

    def test_unclosed_frontmatter(self):
        """Frontmatter with no closing --- should produce an error."""
        issues = audit_skills.check_frontmatter("---\nname: foo\n")
        assert any(i.severity == "error" for i in issues)

    def test_missing_description_field(self):
        """Missing 'description' in frontmatter should warn."""
        content = "---\nname: foo\nversion: 1.0\nauthor: bar\n---\n\nBody"
        issues = audit_skills.check_frontmatter(content)
        assert any("description" in i.message for i in issues)

    def test_description_too_long(self):
        """Description exceeding DESCRIPTION_MAX_LEN should warn."""
        long_desc = "x" * (audit_skills.DESCRIPTION_MAX_LEN + 1)
        content = _make_skill(_BODY_OK, description=long_desc)
        issues = audit_skills.check_frontmatter(content)
        assert any("too long" in i.message.lower() for i in issues)

    def test_description_no_period(self):
        """Description not ending with period should produce an info issue."""
        content = _make_skill(_BODY_OK, description="No period at the end")
        issues = audit_skills.check_frontmatter(content)
        assert any("period" in i.message.lower() for i in issues)


# ── Tests: section checks ─────────────────────────────────────────────────


class TestSections:
    def test_all_sections_present(self):
        """A body with all required sections should produce no issues."""
        issues = audit_skills.check_sections(_make_skill(_BODY_OK))
        assert len(issues) == 0

    def test_missing_sections(self):
        """A body missing required sections should warn for each."""
        content = _make_skill("## Some Random Section\n\nJust content.\n")
        issues = audit_skills.check_sections(content)
        missing = {i.message for i in issues}
        for section in audit_skills.REQUIRED_SECTIONS:
            assert any(section in m for m in missing), (
                f"Should warn about missing '{section}'"
            )

    def test_empty_pitfalls_flagged(self):
        """Pitfalls section with very little content should get an info issue."""
        text = "## Pitfalls\n"
        content = _make_skill(
            _BODY_OK.rsplit("## Pitfalls", 1)[0] + text + "\n## Verification\n## Verification helper\n"
        )
        issues = audit_skills.check_sections(content)
        # The split approach above might be fragile; let's just check
        # that a very short Pitfalls section triggers an info.
        issues = audit_skills.check_sections(
            _make_skill(
                "\n## When to Use\n\n## Prerequisites\n\n## How to Run\n\n## Quick Reference\n\n"
                "## Procedure\n\n## Pitfalls\n\njust a bit\n\n## Verification\n\n"
            )
        )
        assert any(i.severity == "info" for i in issues)


# ── Tests: size check ──────────────────────────────────────────────────────


class TestSize:
    def test_short_skill_warns(self):
        """A skill shorter than MIN_LINES should warn."""
        content = _make_skill("Short\ncontent\n")
        issues = audit_skills.check_size(content)
        assert any("short" in i.message.lower() for i in issues)

    def test_long_skill_warns(self):
        """A skill longer than MAX_LINES should warn."""
        content = _make_skill("\n".join(f"Line {i}" for i in range(audit_skills.MAX_LINES + 5)))
        issues = audit_skills.check_size(content)
        assert any("long" in i.message.lower() for i in issues)

    def test_ideal_range_info(self):
        """A skill in the ideal-ish range should produce no issues."""
        content = _make_skill("\n".join(f"Line {i}" for i in range(60)))
        issues = audit_skills.check_size(content)
        # 60 lines + frontmatter = ~68, which is > 50 (IDEAL_MIN) and < 300 (IDEAL_MAX)
        # So no issues
        assert len([i for i in issues if i.severity in ("error", "warning")]) == 0


# ── Tests: scoring ─────────────────────────────────────────────────────────


class TestScoring:
    def test_perfect_score(self):
        """No issues should yield score 100."""
        assert audit_skills.compute_score([]) == 100

    def test_error_deducts(self):
        """Each error should deduct 15 points."""
        score = audit_skills.compute_score([audit_skills.Issue("error", "Bad")])
        assert score == 85

    def test_warning_deducts(self):
        """Each warning should deduct 5 points."""
        score = audit_skills.compute_score([audit_skills.Issue("warning", "Meh")])
        assert score == 95

    def test_info_deducts(self):
        """Each info should deduct 1 point."""
        score = audit_skills.compute_score([audit_skills.Issue("info", "FYI")])
        assert score == 99

    def test_score_floor(self):
        """Score should not go below 0."""
        score = audit_skills.compute_score(
            [audit_skills.Issue("error", "E") for _ in range(100)]
        )
        assert score == 0

    def test_score_ceiling(self):
        """Score should not go above 100."""
        score = audit_skills.compute_score(
            [audit_skills.Issue("info", "I") for _ in range(-100)]
        )
        assert score == 100


# ── Tests: full audit flow ─────────────────────────────────────────────────


class TestFullAudit:
    def test_audit_valid_skill(self, tmp_path: Path):
        """A well-structured skill should score >= 90."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(_make_skill(_BODY_OK))
        result = audit_skills.audit_skill(skill_dir)
        assert result is not None
        assert result.score >= 90, (
            f"Expected score >= 90 for valid skill, got {result.score}: {result.issues}"
        )

    def test_audit_no_skill_md(self, tmp_path: Path):
        """Directory without SKILL.md should return None."""
        skill_dir = tmp_path / "empty-skill"
        skill_dir.mkdir()
        result = audit_skills.audit_skill(skill_dir)
        assert result is None

    def test_audit_missing_frontmatter(self, tmp_path: Path):
        """Skill with no frontmatter should score low with errors."""
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("Just content, no frontmatter.")
        result = audit_skills.audit_skill(skill_dir)
        assert result is not None
        assert result.score < 50, (
            f"Expected score < 50 for bad skill, got {result.score}"
        )
        assert any("frontmatter" in i["message"].lower() for i in result.issues)


# ── Tests: Issue dataclass ─────────────────────────────────────────────────


class TestIssue:
    def test_issue_creation(self):
        """Issue should store severity and message."""
        issue = audit_skills.Issue("error", "Test error")
        assert issue.severity == "error"
        assert issue.message == "Test error"
