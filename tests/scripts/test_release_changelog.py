"""Tests for commit parsing that feeds the release changelog + attribution.

``scripts/release.py:get_commits`` is the source of truth for the weekly
release notes and the contributor list. It formats ``git log`` with NUL/unit
separators and re-parses the stream; if the record boundary is wrong, the whole
log collapses into a single entry and every commit/author except HEAD is
silently dropped. These tests run ``get_commits`` over a real throwaway repo
and assert it returns one record per commit, with co-authors resolved.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def _load_release_module(monkeypatch, repo_root: Path):
    """Import scripts/release.py with REPO_ROOT pinned to a temp git repo."""
    spec = importlib.util.spec_from_file_location(
        "_release_changelog_under_test",
        Path(__file__).resolve().parents[2] / "scripts" / "release.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    return module


def _commit(repo: Path, name: str, email: str, message: str) -> None:
    """Author a commit with a deterministic, hermetic identity."""
    (repo / "file.txt").write_text(message, encoding="utf-8")
    env = {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
        "GIT_CONFIG_GLOBAL": str(repo / ".gitconfig-empty"),
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    subprocess.run(["git", "add", "."], cwd=repo, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-q", "--no-gpg-sign", "-m", message],
        cwd=repo,
        check=True,
        env=env,
    )


def _init_repo(repo: Path) -> None:
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(repo)], check=True
    )


def test_get_commits_returns_one_record_per_commit(monkeypatch, tmp_path):
    """A multi-commit range must yield every commit, not just HEAD.

    Regression: a "\\0\\0" record split collapsed the whole log into a single
    entry because git never emits two adjacent NULs, so only HEAD survived.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    _commit(repo, "Ada", "ada@example.com", "feat: first thing")
    _commit(repo, "Grace", "grace@example.com", "fix: second thing")
    _commit(
        repo,
        "Linus",
        "linus@example.com",
        "docs: third thing\n\nCo-authored-by: Margaret <margaret@example.com>",
    )

    module = _load_release_module(monkeypatch, repo)
    commits = module.get_commits()

    assert len(commits) == 3
    subjects = [c["subject"] for c in commits]
    assert subjects == [
        "docs: third thing",
        "fix: second thing",
        "feat: first thing",
    ]
    # Every distinct author is preserved, not folded into a single record.
    assert {c["author_name"] for c in commits} == {"Ada", "Grace", "Linus"}
    assert {c["category"] for c in commits} == {"features", "fixes", "docs"}

    # The co-author trailer on the third commit must be parsed from its body.
    head = commits[0]
    assert any(ca == "Margaret" for ca in head["coauthors"])


def test_get_commits_since_tag_excludes_earlier_commits(monkeypatch, tmp_path):
    """A ``tag..HEAD`` range returns only the commits after the tag."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    _commit(repo, "Ada", "ada@example.com", "feat: before tag")
    subprocess.run(["git", "tag", "v2024.01.01"], cwd=repo, check=True)
    _commit(repo, "Grace", "grace@example.com", "fix: after tag one")
    _commit(repo, "Linus", "linus@example.com", "fix: after tag two")

    module = _load_release_module(monkeypatch, repo)
    commits = module.get_commits(since_tag="v2024.01.01")

    assert len(commits) == 2
    assert [c["subject"] for c in commits] == [
        "fix: after tag two",
        "fix: after tag one",
    ]
