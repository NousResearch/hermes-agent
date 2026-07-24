"""Tests for the /review skill slash command contract."""

from pathlib import Path
from shutil import copytree
from unittest.mock import patch

from agent.skill_commands import (
    build_skill_invocation_message,
    resolve_skill_command_key,
    scan_skill_commands,
)

REVIEW_SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "skills" / "software-development" / "review"
)


def _setup_review_skill(tmp_path):
    dest = tmp_path / "software-development" / "review"
    copytree(REVIEW_SKILL_DIR, dest)
    return tmp_path


class TestReviewSkillSlashCommand:
    def test_registers_as_slash_review(self, tmp_path):
        skills_dir = _setup_review_skill(tmp_path)
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            result = scan_skill_commands()

        assert "/review" in result
        assert result["/review"]["name"] == "review"

    def test_resolves_review_command_key(self, tmp_path):
        skills_dir = _setup_review_skill(tmp_path)
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            scan_skill_commands()
            assert resolve_skill_command_key("review") == "/review"

    def test_invocation_preserves_user_instruction(self, tmp_path):
        skills_dir = _setup_review_skill(tmp_path)
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            scan_skill_commands()
            msg = build_skill_invocation_message("/review", "unstaged")

        assert msg is not None
        assert "The user has provided the following instruction" in msg
        assert "unstaged" in msg

    def test_invocation_lists_review_checklist_supporting_file(self, tmp_path):
        skills_dir = _setup_review_skill(tmp_path)
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            scan_skill_commands()
            msg = build_skill_invocation_message("/review")

        assert msg is not None
        assert "supporting files" in msg.lower()
        assert "references/review-checklist.md" in msg
