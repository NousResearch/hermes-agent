"""Every unattended skill mutation leaves a diff behind (#84718 proposal 5).

The reported trace has a background review rewriting a SKILL.md
(`Patched SKILL.md in skill 'hodle-ui-validation' (1 replacement)`) with the
most recent entry under `.curator_backups/` dated eight days earlier — so a
permanent, global policy change inferred from one noisy episode had no diff
and no rollback anywhere on disk.
"""

import json
from unittest.mock import patch

import pytest

from agent.curator_backup import patch_diffs_dir, record_patch_diff
from tools.skill_manager_tool import skill_manage

SKILL = """---
name: reviewed
description: A reviewed skill.
---

# Reviewed

Step 1: capture the surface that changed.
"""


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return skills


def _commit_write(**kwargs):
    """Run skill_manage on the direct-commit path (approval gate bypassed)."""
    return json.loads(skill_manage(**kwargs))


class TestRecordPatchDiff:
    def test_writes_a_unified_diff_with_provenance_header(self, skills_home):
        path = record_patch_diff(
            "hodle-ui-validation", "SKILL.md", "old line\n", "new line\n",
            action="patch",
        )
        assert path is not None
        text = path.read_text(encoding="utf-8")
        assert "# skill: hodle-ui-validation" in text
        assert "# origin: background-review" in text
        assert "-old line" in text and "+new line" in text

    def test_no_change_records_nothing(self, skills_home):
        assert record_patch_diff("s", "SKILL.md", "same\n", "same\n") is None

    def test_oversized_diff_is_truncated(self, skills_home):
        path = record_patch_diff("s", "SKILL.md", "", "x\n" * 200_000)
        assert path is not None
        assert "[diff truncated]" in path.read_text(encoding="utf-8")


class TestAutomatedSkillWriteIsArchived:
    def _patched(self, skills_home, background):
        with patch("tools.skill_manager_tool.SKILLS_DIR", skills_home), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_home]), \
             patch("tools.skill_manager_tool._apply_skill_write_gate", return_value=None):
            assert _commit_write(action="create", name="reviewed", content=SKILL)["success"]
            with patch("tools.skill_provenance.is_background_review", return_value=background), \
                 patch("tools.skill_manager_tool._background_review_write_guard", return_value=None), \
                 patch(
                     "tools.skill_manager_tool._background_review_read_before_write_guard",
                     return_value=None,
                 ):
                return _commit_write(
                    action="patch",
                    name="reviewed",
                    old_string="Step 1: capture the surface that changed.",
                    new_string="Step 1: capture checkout or invoice.",
                )

    def test_background_review_patch_is_archived(self, skills_home):
        result = self._patched(skills_home, background=True)
        assert result["success"] is True
        diffs = sorted(patch_diffs_dir().glob("*.diff"))
        assert len(diffs) == 1
        text = diffs[0].read_text(encoding="utf-8")
        assert "# skill: reviewed" in text
        assert "# action: patch" in text
        assert "-Step 1: capture the surface that changed." in text
        assert "+Step 1: capture checkout or invoice." in text

    def test_foreground_patch_is_not_archived(self, skills_home):
        result = self._patched(skills_home, background=False)
        assert result["success"] is True
        assert not patch_diffs_dir().exists() or not list(patch_diffs_dir().glob("*.diff"))
