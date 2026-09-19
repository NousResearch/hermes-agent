"""Behavioral tests for hermes_cli.build_info — baked build-SHA and git-HEAD helpers."""

import os
import textwrap
from pathlib import Path

import pytest

import hermes_cli.build_info as bi


# ── get_build_sha ─────────────────────────────────────────────────────────────

class TestGetBuildSha:
    def test_returns_none_when_file_absent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bi, "_BUILD_SHA_FILE", tmp_path / "no_such_file")
        assert bi.get_build_sha() is None

    def test_reads_and_truncates_to_8(self, monkeypatch, tmp_path):
        sha_file = tmp_path / ".hermes_build_sha"
        sha_file.write_text("abcdef1234567890" * 2 + "ab" + "\n", encoding="utf-8")
        monkeypatch.setattr(bi, "_BUILD_SHA_FILE", sha_file)
        result = bi.get_build_sha(short=8)
        assert result == "abcdef12"

    def test_short_zero_returns_full_sha(self, monkeypatch, tmp_path):
        full = "a" * 40
        sha_file = tmp_path / ".hermes_build_sha"
        sha_file.write_text(full + "\n", encoding="utf-8")
        monkeypatch.setattr(bi, "_BUILD_SHA_FILE", sha_file)
        assert bi.get_build_sha(short=0) == full

    def test_empty_file_returns_none(self, monkeypatch, tmp_path):
        sha_file = tmp_path / ".hermes_build_sha"
        sha_file.write_text("   \n", encoding="utf-8")
        monkeypatch.setattr(bi, "_BUILD_SHA_FILE", sha_file)
        assert bi.get_build_sha() is None


# ── _resolve_git_head_sha ─────────────────────────────────────────────────────

class TestResolveGitHeadSha:
    def test_regular_checkout(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        refs = git / "refs" / "heads"
        refs.mkdir(parents=True)
        sha = "a" * 40
        (refs / "main").write_text(sha + "\n", encoding="utf-8")
        assert bi._resolve_git_head_sha(tmp_path) == sha

    def test_detached_head(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        sha = "b" * 40
        (git / "HEAD").write_text(sha + "\n", encoding="utf-8")
        assert bi._resolve_git_head_sha(tmp_path) == sha

    def test_packed_refs(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        sha = "c" * 40
        (git / "HEAD").write_text("ref: refs/heads/release\n", encoding="utf-8")
        (git / "packed-refs").write_text(
            f"# pack-refs with: peeled fully-peeled\n{sha} refs/heads/release\n",
            encoding="utf-8",
        )
        assert bi._resolve_git_head_sha(tmp_path) == sha

    def test_no_git_directory_returns_none(self, tmp_path):
        assert bi._resolve_git_head_sha(tmp_path) is None

    def test_worktree_gitfile_pointer(self, tmp_path):
        # Simulate a worktree: .git is a file pointing to a real git dir
        main_git = tmp_path / "main_git"
        main_git.mkdir()
        sha = "d" * 40
        (main_git / "HEAD").write_text("ref: refs/heads/feat\n", encoding="utf-8")
        refs = main_git / "refs" / "heads"
        refs.mkdir(parents=True)
        (refs / "feat").write_text(sha, encoding="utf-8")

        wt = tmp_path / "worktree"
        wt.mkdir()
        (wt / ".git").write_text(f"gitdir: {main_git}\n", encoding="utf-8")
        assert bi._resolve_git_head_sha(wt) == sha


# ── get_code_identity ─────────────────────────────────────────────────────────

def test_get_code_identity_shape(monkeypatch):
    """get_code_identity always returns a dict with the right keys."""
    monkeypatch.setattr(bi, "_code_identity_cache", None)
    result = bi.get_code_identity(refresh=True)
    assert set(result.keys()) == {"sha", "short_sha", "version", "source"}
    assert result["source"] in ("git", "build-file", "unknown")
    if result["sha"] is not None:
        assert result["short_sha"] == result["sha"][:8]


def test_get_code_identity_is_cached(monkeypatch):
    monkeypatch.setattr(bi, "_code_identity_cache", None)
    first = bi.get_code_identity(refresh=True)
    second = bi.get_code_identity()
    assert first == second


def test_get_code_identity_refresh_clears_cache(monkeypatch):
    monkeypatch.setattr(bi, "_code_identity_cache", None)
    bi.get_code_identity(refresh=True)
    assert bi._code_identity_cache is not None
    bi.get_code_identity(refresh=True)  # should not crash
