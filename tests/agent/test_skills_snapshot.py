"""Tests for the fast-path directory-mtime skills snapshot validation."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def skills_dir(tmp_path):
    """Skills directory inside the conftest-provided HERMES_HOME, with one SKILL.md file.

    conftest autouse fixture already set HERMES_HOME = tmp_path / "hermes_test".
    """
    hermes_home = Path(os.environ["HERMES_HOME"])
    sd = hermes_home / "skills"
    sd.mkdir(exist_ok=True)
    skill_subdir = sd / "demo-skill"
    skill_subdir.mkdir(parents=True, exist_ok=True)
    skill_md = "---\nname: demo-skill\ndescription: A demo skill.\n---\n\n# Demo\n\nDoes demo things.\n"
    (skill_subdir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    return sd


def _write_snapshot(skills_dir):
    """Helper: write a real snapshot via the production write path."""
    from agent.prompt_builder import _build_skills_manifest, _write_skills_snapshot

    manifest = _build_skills_manifest(skills_dir)
    _write_skills_snapshot(skills_dir, manifest, [], {})


class TestSkillsSnapshotFastPath:
    def test_fast_path_skips_manifest_rebuild(self, skills_dir):
        """When dir mtime is unchanged, _load_skills_snapshot must NOT call _build_skills_manifest."""
        _write_snapshot(skills_dir)

        from agent import prompt_builder

        with patch.object(
            prompt_builder,
            "_build_skills_manifest",
            wraps=prompt_builder._build_skills_manifest,
        ) as mock_manifest:
            result = prompt_builder._load_skills_snapshot(skills_dir)

        assert result is not None, "Snapshot should have been loaded via fast path"
        mock_manifest.assert_not_called()

    def test_falls_through_on_mtime_change(self, skills_dir):
        """When dir mtime changes, _load_skills_snapshot must fall through to manifest comparison."""
        _write_snapshot(skills_dir)

        # Add a new skill entry inside the skills dir -- this changes the dir mtime.
        new_skill = skills_dir / "new-skill"
        new_skill.mkdir()
        new_skill_md = "---\nname: new-skill\ndescription: New.\n---\n\n# New\n"
        (new_skill / "SKILL.md").write_text(new_skill_md, encoding="utf-8")

        from agent import prompt_builder

        with patch.object(
            prompt_builder,
            "_build_skills_manifest",
            wraps=prompt_builder._build_skills_manifest,
        ) as mock_manifest:
            # Result may be None (manifest mismatch) or a snapshot;
            # what matters is that _build_skills_manifest WAS called.
            prompt_builder._load_skills_snapshot(skills_dir)

        mock_manifest.assert_called_once()

    def test_version_2_rejects_version_1(self, skills_dir):
        """Snapshot with version=1 must be rejected by _load_skills_snapshot."""
        hermes_home = Path(os.environ["HERMES_HOME"])
        snapshot_path = hermes_home / ".skills_prompt_snapshot.json"
        stale = {
            "version": 1,
            "dir_mtime_ns": skills_dir.stat().st_mtime_ns,
            "manifest": {},
            "skills": [],
            "category_descriptions": {},
        }
        snapshot_path.write_text(json.dumps(stale), encoding="utf-8")

        from agent.prompt_builder import _load_skills_snapshot

        result = _load_skills_snapshot(skills_dir)

        assert result is None, "Version-1 snapshot must be rejected"

    def test_write_stores_dir_mtime(self, skills_dir):
        """_write_skills_snapshot must persist dir_mtime_ns as an integer."""
        _write_snapshot(skills_dir)
        hermes_home = Path(os.environ["HERMES_HOME"])
        snapshot_path = hermes_home / ".skills_prompt_snapshot.json"
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))

        assert "dir_mtime_ns" in data, "dir_mtime_ns key must be present in snapshot"
        assert isinstance(data["dir_mtime_ns"], int), (
            f"dir_mtime_ns must be an int, got {type(data['dir_mtime_ns'])}"
        )
