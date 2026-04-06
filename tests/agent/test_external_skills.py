"""Tests for external skill directories (skills.external_dirs config)."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def external_skills_dir(tmp_path):
    """Create a temp dir with a sample external skill."""
    ext_dir = tmp_path / "external-skills"
    skill_dir = ext_dir / "my-external-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-external-skill\ndescription: A skill from an external directory\n---\n\n# My External Skill\n\nDo external things.\n"
    )
    return ext_dir


@pytest.fixture
def hermes_home(tmp_path):
    """Create a minimal HERMES_HOME with config."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    return home


class TestGetExternalSkillsDirs:
    def test_empty_config(self, hermes_home):
        (hermes_home / "config.yaml").write_text("skills:\n  external_dirs: []\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert result == []

    def test_nonexistent_dir_skipped(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "skills:\n  external_dirs:\n    - /nonexistent/path\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert result == []

    def test_valid_dir_returned(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert len(result) == 1
        assert result[0] == external_skills_dir.resolve()

    def test_duplicate_dirs_deduplicated(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n    - {external_skills_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert len(result) == 1

    def test_local_skills_dir_excluded(self, hermes_home):
        local_skills = hermes_home / "skills"
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {local_skills}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert result == []

    def test_no_config_file(self, hermes_home):
        # No config.yaml at all
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert result == []

    def test_string_value_converted_to_list(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs: {external_skills_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert len(result) == 1


class TestGetAllSkillsDirs:
    def test_local_always_first(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_all_skills_dirs
            result = get_all_skills_dirs()
        assert result[0] == hermes_home / "skills"
        assert result[1] == external_skills_dir.resolve()


class TestExternalSkillsInFindAll:
    def test_external_skills_found(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        local_skills = hermes_home / "skills"
        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from tools.skills_tool import _find_all_skills
            skills = _find_all_skills()
        names = [s["name"] for s in skills]
        assert "my-external-skill" in names

    def test_local_takes_precedence(self, hermes_home, external_skills_dir):
        """If the same skill name exists locally and externally, local wins."""
        local_skills = hermes_home / "skills"
        local_skill = local_skills / "my-external-skill"
        local_skill.mkdir(parents=True)
        (local_skill / "SKILL.md").write_text(
            "---\nname: my-external-skill\ndescription: Local version\n---\n\nLocal.\n"
        )
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from tools.skills_tool import _find_all_skills
            skills = _find_all_skills()
        matching = [s for s in skills if s["name"] == "my-external-skill"]
        assert len(matching) == 1
        assert matching[0]["description"] == "Local version"


class TestExternalSkillView:
    def test_skill_view_finds_external(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        local_skills = hermes_home / "skills"
        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from tools.skills_tool import skill_view
            result = json.loads(skill_view("my-external-skill"))
        assert result["success"] is True
        assert "external things" in result["content"]


# ---------------------------------------------------------------------------
# Shared skills (skills.shared config + ~/.hermes/shared-skills/ convention)
# ---------------------------------------------------------------------------


@pytest.fixture
def shared_skills_root(tmp_path, monkeypatch):
    """Create ~/.hermes/shared-skills/ under a fake HOME and return its path."""
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    # Re-import skill_utils so DEFAULT_SHARED_SKILLS picks up the patched HOME.
    # Path.home() is evaluated lazily inside the helper, so we just need the
    # env var to be set before each call.
    shared_root = fake_home / ".hermes" / "shared-skills"
    shared_root.mkdir(parents=True)
    return shared_root


def _make_shared_skill(shared_root, name, description="A shared skill"):
    skill_dir = shared_root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nShared step.\n"
    )
    return skill_dir


class TestGetSharedSkillDirs:
    def test_no_shared_config_returns_empty(self, hermes_home, shared_skills_root, monkeypatch):
        _make_shared_skill(shared_skills_root, "available")
        # config.yaml has no skills.shared key
        (hermes_home / "config.yaml").write_text("skills:\n  external_dirs: []\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # Force-reload the module so DEFAULT_SHARED_SKILLS reflects patched HOME
        import importlib, agent.skill_utils as su
        importlib.reload(su)
        assert su.get_shared_skill_dirs() == []

    def test_shared_skill_resolved_by_name(self, hermes_home, shared_skills_root, monkeypatch):
        _make_shared_skill(shared_skills_root, "team-runbook")
        (hermes_home / "config.yaml").write_text(
            "skills:\n  shared:\n    - team-runbook\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        import importlib, agent.skill_utils as su
        importlib.reload(su)
        result = su.get_shared_skill_dirs()
        assert len(result) == 1
        assert result[0].name == "team-runbook"
        assert result[0].resolve() == (shared_skills_root / "team-runbook").resolve()

    def test_missing_shared_skill_silently_skipped(self, hermes_home, shared_skills_root, monkeypatch):
        _make_shared_skill(shared_skills_root, "exists")
        (hermes_home / "config.yaml").write_text(
            "skills:\n  shared:\n    - exists\n    - does-not-exist\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        import importlib, agent.skill_utils as su
        importlib.reload(su)
        result = su.get_shared_skill_dirs()
        assert len(result) == 1
        assert result[0].name == "exists"

    def test_path_traversal_rejected(self, hermes_home, shared_skills_root, monkeypatch):
        _make_shared_skill(shared_skills_root, "ok")
        (hermes_home / "config.yaml").write_text(
            "skills:\n  shared:\n    - ok\n    - ../../etc\n    - foo/bar\n    - .hidden\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        import importlib, agent.skill_utils as su
        importlib.reload(su)
        result = su.get_shared_skill_dirs()
        # Only "ok" is accepted; the malicious / structural entries are dropped.
        assert len(result) == 1
        assert result[0].name == "ok"

    def test_string_value_converted_to_list(self, hermes_home, shared_skills_root, monkeypatch):
        _make_shared_skill(shared_skills_root, "alone")
        (hermes_home / "config.yaml").write_text(
            "skills:\n  shared: alone\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        import importlib, agent.skill_utils as su
        importlib.reload(su)
        assert len(su.get_shared_skill_dirs()) == 1


class TestSharedInGetAllSkillsDirs:
    def test_shared_appears_after_local_before_external(self, hermes_home, external_skills_dir, shared_skills_root, monkeypatch):
        _make_shared_skill(shared_skills_root, "team-runbook")
        (hermes_home / "config.yaml").write_text(
            "skills:\n"
            "  shared:\n    - team-runbook\n"
            f"  external_dirs:\n    - {external_skills_dir}\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        import importlib, agent.skill_utils as su
        importlib.reload(su)
        dirs = su.get_all_skills_dirs()
        # Order: local, shared, external
        assert dirs[0] == hermes_home / "skills"
        assert dirs[1].resolve() == (shared_skills_root / "team-runbook").resolve()
        assert dirs[2].resolve() == external_skills_dir.resolve()

    def test_no_shared_config_backward_compat(self, hermes_home, external_skills_dir, shared_skills_root, monkeypatch):
        # No skills.shared key — must behave exactly as before this PR
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        import importlib, agent.skill_utils as su
        importlib.reload(su)
        dirs = su.get_all_skills_dirs()
        assert len(dirs) == 2
        assert dirs[0] == hermes_home / "skills"
        assert dirs[1].resolve() == external_skills_dir.resolve()

    def test_shared_not_duplicated_when_also_in_external_dirs(self, hermes_home, shared_skills_root, monkeypatch):
        skill_dir = _make_shared_skill(shared_skills_root, "team-runbook")
        # User points external_dirs at the same individual skill dir AND
        # references it via skills.shared. Should only appear once.
        (hermes_home / "config.yaml").write_text(
            "skills:\n"
            "  shared:\n    - team-runbook\n"
            f"  external_dirs:\n    - {skill_dir}\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        import importlib, agent.skill_utils as su
        importlib.reload(su)
        dirs = su.get_all_skills_dirs()
        resolved = [d.resolve() for d in dirs]
        assert resolved.count(skill_dir.resolve()) == 1


class TestSelectiveSharingAcrossProfiles:
    """Two profiles with different skills.shared lists each see their own subset."""

    def test_profile_a_and_b_see_different_subsets(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        shared = fake_home / ".hermes" / "shared-skills"
        shared.mkdir(parents=True)
        _make_shared_skill(shared, "tool-one")
        _make_shared_skill(shared, "tool-two")
        _make_shared_skill(shared, "tool-three")

        profile_a = tmp_path / "profile-a"
        (profile_a / "skills").mkdir(parents=True)
        (profile_a / "config.yaml").write_text(
            "skills:\n  shared:\n    - tool-one\n    - tool-two\n"
        )
        profile_b = tmp_path / "profile-b"
        (profile_b / "skills").mkdir(parents=True)
        (profile_b / "config.yaml").write_text(
            "skills:\n  shared:\n    - tool-three\n"
        )

        import importlib, agent.skill_utils as su

        monkeypatch.setenv("HERMES_HOME", str(profile_a))
        importlib.reload(su)
        a_dirs = [d.name for d in su.get_all_skills_dirs()]
        assert "tool-one" in a_dirs
        assert "tool-two" in a_dirs
        assert "tool-three" not in a_dirs

        monkeypatch.setenv("HERMES_HOME", str(profile_b))
        importlib.reload(su)
        b_dirs = [d.name for d in su.get_all_skills_dirs()]
        assert "tool-three" in b_dirs
        assert "tool-one" not in b_dirs
        assert "tool-two" not in b_dirs


class TestLocalShadowsShared:
    def test_profile_local_skill_takes_precedence_over_shared_by_position(
        self, hermes_home, shared_skills_root, monkeypatch,
    ):
        # Same skill name in profile-local and shared.
        # get_all_skills_dirs returns local first, so any consumer that
        # iterates in order will pick up the local copy.
        (hermes_home / "skills").mkdir(exist_ok=True)
        local = hermes_home / "skills" / "clash"
        local.mkdir()
        (local / "SKILL.md").write_text(
            "---\nname: clash\ndescription: local version\n---\n\n# local\n"
        )
        _make_shared_skill(shared_skills_root, "clash", "shared version")
        (hermes_home / "config.yaml").write_text(
            "skills:\n  shared:\n    - clash\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        import importlib, agent.skill_utils as su
        importlib.reload(su)
        dirs = su.get_all_skills_dirs()
        # Local first, shared second
        assert dirs[0].resolve() == (hermes_home / "skills").resolve()
        assert dirs[1].resolve() == (shared_skills_root / "clash").resolve()
