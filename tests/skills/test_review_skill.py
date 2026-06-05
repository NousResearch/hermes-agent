"""Tests for the /review skill — frontmatter validation, slash command registration,
and skill_view loading."""

from pathlib import Path
from unittest.mock import patch

from tools.skills_tool import _parse_frontmatter, skill_matches_platform
from agent.skill_commands import (
    build_skill_invocation_message,
    resolve_skill_command_key,
    scan_skill_commands,
)

REVIEW_SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "review"
    / "SKILL.md"
)


def _read_skill():
    return REVIEW_SKILL_PATH.read_text(encoding="utf-8")


class TestReviewSkillFrontmatter:
    def test_skill_file_exists(self):
        assert REVIEW_SKILL_PATH.exists(), "SKILL.md not found at expected path"

    def test_frontmatter_has_required_fields(self):
        fm, body = _parse_frontmatter(_read_skill())
        assert fm.get("name") == "review"
        assert fm.get("description"), "description must not be empty"
        assert len(fm["description"]) <= 1024

    def test_frontmatter_has_version(self):
        fm, _ = _parse_frontmatter(_read_skill())
        assert "version" in fm

    def test_frontmatter_has_tags(self):
        fm, _ = _parse_frontmatter(_read_skill())
        tags = fm.get("metadata", {}).get("hermes", {}).get("tags", [])
        assert isinstance(tags, list)
        assert len(tags) > 0

    def test_frontmatter_has_related_skills(self):
        fm, _ = _parse_frontmatter(_read_skill())
        related = fm.get("metadata", {}).get("hermes", {}).get("related_skills", [])
        assert isinstance(related, list)
        assert "requesting-code-review" in related
        assert "github-code-review" in related

    def test_no_platform_restriction(self):
        fm, _ = _parse_frontmatter(_read_skill())
        assert "platforms" not in fm, "/review should work on all platforms"

    def test_platform_match_returns_true(self):
        fm, _ = _parse_frontmatter(_read_skill())
        assert skill_matches_platform(fm) is True

    def test_body_is_not_empty(self):
        _, body = _parse_frontmatter(_read_skill())
        assert len(body.strip()) > 100, "skill body too short"


class TestReviewSkillContent:
    def test_documents_all_four_modes(self):
        content = _read_skill()
        assert "staged" in content.lower()
        assert "unstaged" in content.lower()
        assert "file" in content.lower()
        assert "pr" in content.lower()

    def test_documents_invocation_syntax(self):
        content = _read_skill()
        assert "/review" in content
        assert "/review unstaged" in content
        assert "/review pr" in content

    def test_references_directory_exists(self):
        references_dir = REVIEW_SKILL_PATH.parent / "references"
        assert references_dir.is_dir()

    def test_review_checklist_exists(self):
        checklist = REVIEW_SKILL_PATH.parent / "references" / "review-checklist.md"
        assert checklist.exists()
        content = checklist.read_text(encoding="utf-8")
        assert "Correctness" in content
        assert "Security" in content


class TestReviewSkillSlashCommand:
    def _setup_skills_dir(self, tmp_path):
        """Copy the real review skill into a temp skills dir for scanning."""
        dest = tmp_path / "software-development" / "review"
        dest.mkdir(parents=True)
        skill_content = _read_skill()
        (dest / "SKILL.md").write_text(skill_content, encoding="utf-8")
        return tmp_path

    def test_registers_as_slash_review(self, tmp_path):
        skills_dir = self._setup_skills_dir(tmp_path)
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            result = scan_skill_commands()
        assert "/review" in result
        assert result["/review"]["name"] == "review"

    def test_resolve_command_key(self, tmp_path):
        skills_dir = self._setup_skills_dir(tmp_path)
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            scan_skill_commands()
            assert resolve_skill_command_key("review") == "/review"

    def test_underscore_form_resolves(self, tmp_path):
        skills_dir = self._setup_skills_dir(tmp_path)
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            scan_skill_commands()
            assert resolve_skill_command_key("review") == "/review"

    def test_build_invocation_message(self, tmp_path):
        skills_dir = self._setup_skills_dir(tmp_path)
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            scan_skill_commands()
            msg = build_skill_invocation_message("/review", "unstaged")
        assert msg is not None
        assert "review" in msg.lower()
        assert "unstaged" in msg

    def test_build_invocation_with_file_arg(self, tmp_path):
        skills_dir = self._setup_skills_dir(tmp_path)
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            scan_skill_commands()
            msg = build_skill_invocation_message("/review", "src/main.py")
        assert msg is not None
        assert "src/main.py" in msg

    def test_build_invocation_no_args(self, tmp_path):
        skills_dir = self._setup_skills_dir(tmp_path)
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            scan_skill_commands()
            msg = build_skill_invocation_message("/review")
        assert msg is not None
        assert "review" in msg.lower()

    def test_supporting_files_hint_present(self, tmp_path):
        skills_dir = self._setup_skills_dir(tmp_path)
        refs = skills_dir / "software-development" / "review" / "references"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "review-checklist.md").write_text("checklist content")
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            scan_skill_commands()
            msg = build_skill_invocation_message("/review")
        assert msg is not None
        assert "supporting files" in msg.lower() or "skill_view" in msg
