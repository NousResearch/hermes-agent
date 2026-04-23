"""Tests for project-local skill discovery and runtime overlay ordering."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_home(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    return home


class TestRuntimeCwdAndGitRoot:
    def test_find_git_root_walks_parents(self, tmp_path):
        repo = tmp_path / "repo"
        nested = repo / "a" / "b"
        (repo / ".git").mkdir(parents=True)
        nested.mkdir(parents=True)

        from agent.skill_utils import find_git_root

        assert find_git_root(nested) == repo.resolve()

    def test_find_git_root_returns_none_outside_repo(self, tmp_path):
        isolated = tmp_path / "isolated"
        isolated.mkdir()

        from agent.skill_utils import find_git_root

        assert find_git_root(isolated) is None

    def test_resolve_runtime_cwd_prefers_terminal_cwd(self, tmp_path, monkeypatch):
        env_cwd = tmp_path / "env-cwd"
        env_cwd.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(env_cwd))

        from agent.skill_utils import resolve_runtime_cwd

        assert resolve_runtime_cwd() == env_cwd.resolve()


class TestProjectLocalSkillsDirs:
    def test_project_skill_dirs_prefer_dot_hermes_before_dot_agents(self, tmp_path):
        repo = tmp_path / "repo"
        cwd = repo / "src"
        (repo / ".git").mkdir(parents=True)
        (repo / ".hermes" / "skills").mkdir(parents=True)
        (repo / ".agents" / "skills").mkdir(parents=True)
        cwd.mkdir(parents=True)

        from agent.skill_utils import get_project_local_skills_dirs

        assert get_project_local_skills_dirs(cwd) == [
            (repo / ".hermes" / "skills").resolve(),
            (repo / ".agents" / "skills").resolve(),
        ]

    def test_project_skill_dirs_nearest_ancestor_first(self, tmp_path):
        repo = tmp_path / "repo"
        cwd = repo / "packages" / "app" / "src"
        (repo / ".git").mkdir(parents=True)
        (repo / ".hermes" / "skills").mkdir(parents=True)
        (repo / ".agents" / "skills").mkdir(parents=True)
        (repo / "packages" / "app" / ".hermes" / "skills").mkdir(parents=True)
        (repo / "packages" / "app" / ".agents" / "skills").mkdir(parents=True)
        cwd.mkdir(parents=True)

        from agent.skill_utils import get_project_local_skills_dirs

        assert get_project_local_skills_dirs(cwd) == [
            (repo / "packages" / "app" / ".hermes" / "skills").resolve(),
            (repo / "packages" / "app" / ".agents" / "skills").resolve(),
            (repo / ".hermes" / "skills").resolve(),
            (repo / ".agents" / "skills").resolve(),
        ]

    def test_project_skill_dirs_without_git_root_only_use_cwd(self, tmp_path):
        cwd = tmp_path / "workspace"
        parent = tmp_path
        (cwd / ".hermes" / "skills").mkdir(parents=True)
        (parent / ".agents" / "skills").mkdir(parents=True)

        from agent.skill_utils import get_project_local_skills_dirs

        assert get_project_local_skills_dirs(cwd) == [
            (cwd / ".hermes" / "skills").resolve(),
        ]


class TestRuntimeSkillsDirs:
    def test_runtime_skills_dirs_project_local_then_home_then_external(self, hermes_home, tmp_path):
        repo = tmp_path / "repo"
        cwd = repo / "src"
        external = tmp_path / "external-skills"
        external.mkdir(parents=True)
        (repo / ".git").mkdir(parents=True)
        (repo / ".hermes" / "skills").mkdir(parents=True)
        cwd.mkdir(parents=True)
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external}\n"
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_runtime_skills_dirs

            assert get_runtime_skills_dirs(cwd) == [
                (repo / ".hermes" / "skills").resolve(),
                (hermes_home / "skills").resolve(),
                external.resolve(),
            ]
