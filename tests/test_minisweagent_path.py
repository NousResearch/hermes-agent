"""Tests for minisweagent_path.py."""

from pathlib import Path

from minisweagent_path import discover_minisweagent_src


def test_discover_minisweagent_src_in_current_checkout(tmp_path):
    repo = tmp_path / "repo"
    src = repo / "mini-swe-agent" / "src"
    src.mkdir(parents=True)

    assert discover_minisweagent_src(repo) == src.resolve()


def test_discover_minisweagent_src_falls_back_from_worktree_to_main_checkout(tmp_path):
    main_repo = tmp_path / "main-repo"
    (main_repo / ".git" / "worktrees" / "wt1").mkdir(parents=True)
    main_src = main_repo / "mini-swe-agent" / "src"
    main_src.mkdir(parents=True)

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {main_repo / '.git' / 'worktrees' / 'wt1'}\n", encoding="utf-8")
    (worktree / "mini-swe-agent").mkdir()  # empty placeholder, no src/

    assert discover_minisweagent_src(worktree) == main_src.resolve()


def test_discover_minisweagent_src_returns_none_when_missing(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    assert discover_minisweagent_src(repo) is None
