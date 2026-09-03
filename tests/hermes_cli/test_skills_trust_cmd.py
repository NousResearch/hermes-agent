"""`hermes skills trust` folds linked-worktree paths to the repo root.

Trusting a worktree path would accrete per-worktree entries that die with
`git worktree remove`; trust is repo-level, so the command stores the
common root instead (agent/skill_utils._linked_worktree_common_root).
"""

import argparse
from pathlib import Path

import pytest
import yaml


def _make_linked_worktree(tmp_path, name="wt"):
    """Repo + linked worktree laid out as `git worktree add` does (file-only,
    no git binary needed — same fixture as tests/agent/test_project_skills.py)."""
    repo = tmp_path / "repo"
    admin = repo / ".git" / "worktrees" / name
    admin.mkdir(parents=True)
    (admin / "commondir").write_text("../..\n")
    wt = tmp_path / name
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {admin}\n")
    return repo, wt


@pytest.fixture
def cli_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    (home / "config.yaml").write_text("skills:\n  external_dirs: []\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    import agent.skill_utils as su

    su._external_dirs_cache_clear()
    yield home
    su._external_dirs_cache_clear()


def _trusted_dirs(home: Path):
    cfg = yaml.safe_load((home / "config.yaml").read_text()) or {}
    return [str(p) for p in (cfg.get("skills", {}).get("trusted_project_dirs") or [])]


def _run_trust(path=None, action="trust"):
    from hermes_cli.main import _cmd_skills_trust

    args = argparse.Namespace(skills_action=action, path=str(path) if path else None)
    _cmd_skills_trust(args)


class TestSkillsTrustWorktreeFolding:
    def test_explicit_worktree_path_stores_common_root(
        self, cli_home, tmp_path, capsys
    ):
        repo, wt = _make_linked_worktree(tmp_path)
        _run_trust(path=wt)
        assert _trusted_dirs(cli_home) == [str(repo.resolve())]
        out = capsys.readouterr().out
        assert "worktree" in out.lower()

    def test_cwd_inside_worktree_stores_common_root(
        self, cli_home, tmp_path, monkeypatch, capsys
    ):
        repo, wt = _make_linked_worktree(tmp_path)
        monkeypatch.chdir(wt)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        _run_trust()
        assert _trusted_dirs(cli_home) == [str(repo.resolve())]

    def test_plain_repo_unchanged(self, cli_home, tmp_path, capsys):
        repo = tmp_path / "plain"
        (repo / ".git").mkdir(parents=True)
        _run_trust(path=repo)
        assert _trusted_dirs(cli_home) == [str(repo.resolve())]

    def test_worktree_of_already_trusted_repo_reports_already_trusted(
        self, cli_home, tmp_path, capsys
    ):
        repo, wt = _make_linked_worktree(tmp_path)
        _run_trust(path=repo)
        capsys.readouterr()
        _run_trust(path=wt)
        assert _trusted_dirs(cli_home) == [str(repo.resolve())]
        assert "Already trusted" in capsys.readouterr().out

    def test_untrust_via_worktree_path_removes_repo_entry(
        self, cli_home, tmp_path, capsys
    ):
        repo, wt = _make_linked_worktree(tmp_path)
        _run_trust(path=repo)
        _run_trust(path=wt, action="untrust")
        assert _trusted_dirs(cli_home) == []
