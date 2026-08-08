"""Regression tests for the s1-w1b extraction of the git-worktree cluster
(cluster c10) from cli.py into ``hermes_cli/worktree_mixin.py``.

Verifies byte-fidelity of the move: the mixin module owns the moved
functions, and cli.py still re-exports them so the ``from cli import ...``
API (hermes_cli/main.py, tests/cli/test_worktree_security.py,
tests/tools/test_windows_native_support.py) is unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

WORKTREE_NAMES = [
    "_active_worktree",
    "_WORKTREE_MERGE_CACHE_MAX",
    "_cleanup_worktree",
    "_git_repo_root",
    "_load_worktree_merge_cache",
    "_normalize_git_bash_path",
    "_path_is_within_root",
    "_prune_orphaned_branches",
    "_prune_stale_worktrees",
    "_resolve_worktree_base",
    "_save_worktree_merge_cache",
    "_setup_worktree",
    "_worktree_commits_all_merged_upstream",
    "_worktree_has_unpushed_commits",
    "_worktree_is_dirty",
    "_worktree_lock_is_live",
    "_worktree_merge_cache_path",
]


def test_mixin_reexports_worktree_cluster():
    import hermes_cli.worktree_mixin as wt

    for name in WORKTREE_NAMES:
        assert hasattr(wt, name), name


def test_cli_still_reexports_worktree_cluster():
    """cli.py re-imports every moved name; identity must be preserved so
    the module-level state and functions stay the same objects."""
    import cli as cli_mod
    import hermes_cli.worktree_mixin as wt

    for name in WORKTREE_NAMES:
        assert getattr(cli_mod, name) is getattr(wt, name), name


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/c/Users/andre/repo", "C:\\Users\\andre\\repo"),
        ("/C/Users/andre/repo", "C:\\Users\\andre\\repo"),
        ("/cygdrive/c/Users/andre", "C:\\Users\\andre"),
        ("/mnt/c/Users/andre", "C:\\Users\\andre"),
        ("C:/Users/andre/repo", "C:/Users/andre/repo"),
        ("C:\\Users\\andre\\repo", "C:\\Users\\andre\\repo"),
        (None, None),
        ("", ""),
    ],
)
def test_normalize_git_bash_path(monkeypatch, path, expected):
    from hermes_cli.worktree_mixin import _normalize_git_bash_path

    monkeypatch.setattr(sys, "platform", "win32")
    assert _normalize_git_bash_path(path) == expected


def test_normalize_git_bash_path_non_windows_is_noop(monkeypatch):
    from hermes_cli.worktree_mixin import _normalize_git_bash_path

    monkeypatch.setattr(sys, "platform", "linux")
    assert _normalize_git_bash_path("/c/Users/andre") == "/c/Users/andre"


def test_path_is_within_root():
    from hermes_cli.worktree_mixin import _path_is_within_root

    root = Path("/repo")
    assert _path_is_within_root(Path("/repo/a/b"), root) is True
    assert _path_is_within_root(Path("/repo"), root) is True
    assert _path_is_within_root(Path("/other"), root) is False
    assert _path_is_within_root(Path("/repo2"), root) is False


def test_worktree_merge_cache_path_under_hermes_home():
    from hermes_constants import get_hermes_home
    from hermes_cli.worktree_mixin import _worktree_merge_cache_path

    p = _worktree_merge_cache_path()
    assert isinstance(p, Path)
    assert p == get_hermes_home() / "cache" / "worktree_merge_verdicts.json"


def test_load_merge_cache_missing_file_is_empty(tmp_path, monkeypatch):
    from hermes_cli.worktree_mixin import _load_worktree_merge_cache, _worktree_merge_cache_path

    monkeypatch.setattr(
        "hermes_cli.worktree_mixin._worktree_merge_cache_path",
        lambda: tmp_path / "does-not-exist.json",
    )
    assert _load_worktree_merge_cache() == {}


def test_save_and_load_merge_cache_roundtrip(tmp_path, monkeypatch):
    from hermes_cli.worktree_mixin import (
        _load_worktree_merge_cache,
        _save_worktree_merge_cache,
        _worktree_merge_cache_path,
    )

    cache_file = tmp_path / "verdicts.json"
    monkeypatch.setattr(
        "hermes_cli.worktree_mixin._worktree_merge_cache_path",
        lambda: cache_file,
    )
    _save_worktree_merge_cache({"a..b:20": True, "c..d:20": False})
    assert _load_worktree_merge_cache() == {"a..b:20": True, "c..d:20": False}


def test_save_merge_cache_rejects_non_bool_entries(tmp_path, monkeypatch):
    """Corrupt/foreign cache entries must never inject non-bools."""
    from hermes_cli.worktree_mixin import _load_worktree_merge_cache, _worktree_merge_cache_path

    cache_file = tmp_path / "verdicts.json"
    cache_file.write_text('{"version": 1, "verdicts": {"ok": true, "evil": "yes"}}', encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.worktree_mixin._worktree_merge_cache_path",
        lambda: cache_file,
    )
    assert _load_worktree_merge_cache() == {"ok": True}
