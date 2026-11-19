#!/usr/bin/env python3
"""Tests for the changelog-gen optional skill helper.

Pure-function coverage for parsing/categorization plus an end-to-end CLI check
against a throwaway git repo exercising multiline-body records, the NUL/RS
stream split, JSON-exclusive mode, and per-tag grouping. Stdlib + pytest only,
no network.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_SCRIPT = os.path.join(
    HERE, "..", "..", "optional-skills", "research",
    "changelog-gen", "scripts", "changelog_gen.py",
)
SKILL_SCRIPT = os.path.normpath(SKILL_SCRIPT)

sys.path.insert(0, os.path.dirname(SKILL_SCRIPT))

import changelog_gen as cg  # noqa: E402


# --- NUL/RS record parsing (defect: line 57/61) ---

def test_parse_commit_records_triplet():
    raw = (
        b"abc123\x00feat: add thing\x00body line 1\nbody line 2\x1e"
        b"def456\x00fix(core): repair bug\x00\x1e"
    )
    out = cg.parse_commit_records(raw)
    assert len(out) == 2
    assert out[0]["sha"] == "abc123"
    assert out[0]["subject"] == "feat: add thing"
    # Multiline body survives intact (no field-splitting inside a record).
    assert out[0]["body"] == "body line 1\nbody line 2"
    assert out[1]["sha"] == "def456"
    assert out[1]["subject"] == "fix(core): repair bug"
    assert out[1]["body"] == ""


def test_parse_commit_records_skips_blanks():
    raw = b"\x00\x00\x1e\x00\x00\x1eabc123\x00feat: x\x00\x1e\x1e"
    out = cg.parse_commit_records(raw)
    assert len(out) == 1
    assert out[0]["sha"] == "abc123"


def test_get_commits_returns_commits(tmp_git_repo):
    """get_commits must parse the git stream into >=1 commit (no 'No commits')."""
    commits = cg.get_commits(tmp_git_repo, all_commits=True)
    assert len(commits) >= 1
    c = commits[0]
    assert set(c.keys()) == {"sha", "subject", "body"}
    assert len(c["sha"]) == 40  # full hash from git


# --- categorize (defect-free but regression-critical) ---

def test_categorize_groups_by_type():
    commits = [
        {"sha": "a", "subject": "feat: a", "body": ""},
        {"sha": "b", "subject": "fix: b", "body": ""},
        {"sha": "c", "subject": "random message", "body": ""},
    ]
    cat = cg.categorize_commits(commits)
    assert len(cat["feat"]) == 1
    assert len(cat["fix"]) == 1
    assert len(cat["other"]) == 1


def test_parse_commit_breaking():
    p = cg.parse_commit("feat(api)!: drop endpoint", "")
    assert p["breaking"] is True
    assert p["type"] == "breaking"
    p2 = cg.parse_commit("fix: normal", "BREAKING CHANGE: removed flag")
    assert p2["breaking"] is True


def test_parse_commit_scope():
    p = cg.parse_commit("refactor(core): tidy", "")
    assert p["scope"] == "core"
    assert p["description"] == "tidy"


# --- JSON exclusive mode (defect: line 180) ---

def test_json_exclusive_no_markdown(tmp_git_repo, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["changelog_gen.py", "--path", tmp_git_repo, "--json"])
    cg.main()
    out = capsys.readouterr().out
    data = json.loads(out)  # must be valid, parseable JSON on its own
    assert data["total_commits"] >= 1
    assert "repo" in data


def test_json_stats_counts_by_type(tmp_git_repo):
    commits = cg.get_commits(tmp_git_repo, all_commits=True)
    cat = cg.categorize_commits(commits)
    stats = cg._build_stats("repo", commits, cat, [])
    assert stats["total_commits"] == len(commits)
    # Every counted type has count >= 1.
    assert all(v >= 1 for v in stats["by_type"].values())


# --- per-tag grouping (defect: line 143) ---

def test_get_tag_groups_assigns_commits(tmp_git_repo_tags):
    repo, tag_names = tmp_git_repo_tags
    commits = cg.get_commits(repo, all_commits=True)
    groups = cg.get_tag_groups(repo, all_commits=True, commits=commits)
    labels = [label for label, _ in groups]
    # Unreleased plus each tag plus a "Before <oldest>" tail.
    assert "Unreleased" in labels
    assert any(label.startswith("Before ") for label in labels)
    # At least one bucket actually holds a commit.
    total_assigned = sum(len(c) for _, c in groups)
    assert total_assigned >= 1


def test_format_changelog_uses_tag_groups():
    commits = [
        {"sha": "aa", "subject": "feat: x", "body": ""},
        {"sha": "bb", "subject": "fix: y", "body": ""},
    ]
    cat = cg.categorize_commits(commits)
    out = cg.format_changelog(cat, "repo", tags=["v1"], tag_groups=[("Unreleased", commits)])
    assert "## Unreleased" in out
    assert "### Features" in out
    assert "### Bug Fixes" in out


# --- fixtures ---

@pytest.fixture
def tmp_git_repo():
    d = tempfile.mkdtemp()
    _run(d, ["git", "init", "-q"])
    _run(d, ["git", "config", "user.email", "test@example.com"])
    _run(d, ["git", "config", "user.name", "Test"])
    _write(d, "f.txt", "hello")
    _run(d, ["git", "add", "f.txt"])
    _run(d, ["git", "commit", "-q", "-m", "feat: initial commit\n\nMultiline body\nsecond line."])
    yield d
    _rmtree(d)


@pytest.fixture
def tmp_git_repo_tags():
    d = tempfile.mkdtemp()
    _run(d, ["git", "init", "-q"])
    _run(d, ["git", "config", "user.email", "test@example.com"])
    _run(d, ["git", "config", "user.name", "Test"])
    _write(d, "a.txt", "1")
    _run(d, ["git", "add", "a.txt"])
    _run(d, ["git", "commit", "-q", "-m", "feat: v1 work"])
    _run(d, ["git", "tag", "v1.0.0"])
    _write(d, "b.txt", "2")
    _run(d, ["git", "add", "b.txt"])
    _run(d, ["git", "commit", "-q", "-m", "fix: v1.1 fix"])
    _run(d, ["git", "tag", "v1.1.0"])
    _write(d, "c.txt", "3")
    _run(d, ["git", "add", "c.txt"])
    _run(d, ["git", "commit", "-q", "-m", "feat: unreleased work"])
    yield d, ["v1.0.0", "v1.1.0"]
    _rmtree(d)


# --- helpers ---

def _run(cwd, cmd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


def _write(cwd, name, content):
    with open(os.path.join(cwd, name), "w") as f:
        f.write(content)


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)
