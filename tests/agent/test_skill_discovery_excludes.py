"""Test skill discovery excludes .bak-* and backup directories.

Covers: https://github.com/NousResearch/hermes-agent/issues/25113

The skill loader must not discover SKILL.md files inside directories
whose names contain backup markers (.bak, .backup, backup-).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# _is_excluded_skill_dir unit tests
# ---------------------------------------------------------------------------

class TestIsExcludedSkillDir:
    """Verify backup-directory detection logic."""

    def test_exact_excluded_dirs(self):
        """Well-known excluded directory names."""
        from agent.skill_utils import _is_excluded_skill_dir
        for name in (".git", ".github", ".hub", ".archive", "__pycache__"):
            assert _is_excluded_skill_dir(name) is True, f"{name} should be excluded"

    def test_normal_skill_dirs_not_excluded(self):
        """Regular skill directory names must not be excluded."""
        from agent.skill_utils import _is_excluded_skill_dir
        for name in ("my-skill", "hermes-profile-convergence", "daily-stock-report", "devops"):
            assert _is_excluded_skill_dir(name) is False, f"{name} should NOT be excluded"

    def test_bak_suffix_excluded(self):
        """Directories with .bak-<timestamp> pattern are excluded."""
        from agent.skill_utils import _is_excluded_skill_dir
        for name in (
            "hermes-profile-convergence.bak-20260510_234206",
            "hermes-agent.bak-20260510_233500",
            "my-skill.bak-v2",
            "something.bak",
        ):
            assert _is_excluded_skill_dir(name) is True, f"{name} should be excluded"

    def test_backup_prefix_excluded(self):
        """Directories starting with .backup- or backup- are excluded."""
        from agent.skill_utils import _is_excluded_skill_dir
        for name in (
            "my-skill.backup-v2",
            ".backup-20260510",
            "backup-old-skill",
        ):
            assert _is_excluded_skill_dir(name) is True, f"{name} should be excluded"

    def test_case_insensitive(self):
        """Backup detection is case-insensitive."""
        from agent.skill_utils import _is_excluded_skill_dir
        assert _is_excluded_skill_dir("skill.BAK-timestamp") is True
        assert _is_excluded_skill_dir("skill.Backup-v2") is True


# ---------------------------------------------------------------------------
# iter_skill_index_files integration tests
# ---------------------------------------------------------------------------

class TestIterSkillIndexFiles:
    """Verify iter_skill_index_files skips backup directories."""

    def test_skips_bak_directories(self, tmp_path):
        """.bak-* directories are not yielded."""
        from agent.skill_utils import iter_skill_index_files

        # Create a live skill
        live = tmp_path / "my-skill"
        live.mkdir()
        (live / "SKILL.md").write_text("---\nname: my-skill\n---\nLive content")

        # Create a backup skill (should be skipped)
        bak = tmp_path / "my-skill.bak-20260510"
        bak.mkdir()
        (bak / "SKILL.md").write_text("---\nname: my-skill\n---\nStale content")

        results = list(iter_skill_index_files(tmp_path, "SKILL.md"))
        assert len(results) == 1
        assert "my-skill.bak" not in str(results[0])

    def test_skips_pycache(self, tmp_path):
        """__pycache__ directories are not yielded."""
        from agent.skill_utils import iter_skill_index_files

        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "SKILL.md").write_text("---\nname: bad\n---\n")

        live = tmp_path / "good-skill"
        live.mkdir()
        (live / "SKILL.md").write_text("---\nname: good\n---\n")

        results = list(iter_skill_index_files(tmp_path, "SKILL.md"))
        assert len(results) == 1
        assert results[0].parent.name == "good-skill"

    def test_nested_bak_excluded(self, tmp_path):
        """Nested .bak-* directories are excluded even under valid category dirs."""
        from agent.skill_utils import iter_skill_index_files

        cat = tmp_path / "devops"
        cat.mkdir()

        live = cat / "hermes-backup"
        live.mkdir()
        (live / "SKILL.md").write_text("---\nname: hermes-backup\n---\nLive")

        bak = cat / "hermes-backup.bak-20260510"
        bak.mkdir()
        (bak / "SKILL.md").write_text("---\nname: hermes-backup\n---\nStale")

        results = list(iter_skill_index_files(tmp_path, "SKILL.md"))
        assert len(results) == 1
        assert "bak-" not in str(results[0])
