"""
tests/test_git_intel.py — Unit tests for the git_intel toolset

Tests use a temporary git repository created in a temp directory so
no real repo is required. All tests use pure stdlib only.
"""

import os
import subprocess
import tempfile
import shutil
import pytest

from tools.git_intel_tool import (
    git_repo_summary,
    git_log,
    git_diff_stats,
    git_contributors,
    git_file_history,
    git_branch_compare,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def temp_repo():
    """Create a minimal git repo for testing and clean it up after."""
    tmpdir = tempfile.mkdtemp()
    env = {**os.environ, "GIT_AUTHOR_NAME": "Test User", "GIT_AUTHOR_EMAIL": "test@example.com",
           "GIT_COMMITTER_NAME": "Test User", "GIT_COMMITTER_EMAIL": "test@example.com"}

    def run(*cmd):
        subprocess.run(list(cmd), cwd=tmpdir, env=env, capture_output=True, check=True)

    run("git", "init")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test User")

    # Initial commit
    with open(os.path.join(tmpdir, "README.md"), "w") as f:
        f.write("# Test Repo\n")
    run("git", "add", ".")
    run("git", "commit", "-m", "initial commit")

    # Second commit
    with open(os.path.join(tmpdir, "main.py"), "w") as f:
        f.write("print('hello')\n" * 10)
    run("git", "add", ".")
    run("git", "commit", "-m", "add main.py")

    # Third commit — modify README
    with open(os.path.join(tmpdir, "README.md"), "a") as f:
        f.write("More content\n" * 5)
    run("git", "add", ".")
    run("git", "commit", "-m", "update README")

    yield tmpdir
    shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# git_repo_summary
# ---------------------------------------------------------------------------

class TestGitRepoSummary:
    def test_returns_dict(self, temp_repo):
        result = git_repo_summary(temp_repo)
        assert isinstance(result, dict)
        assert "error" not in result

    def test_has_expected_keys(self, temp_repo):
        result = git_repo_summary(temp_repo)
        for key in ("current_branch", "branches", "total_commits", "tracked_files",
                    "contributors_count", "latest_commit", "repo_path"):
            assert key in result, f"Missing key: {key}"

    def test_total_commits(self, temp_repo):
        result = git_repo_summary(temp_repo)
        assert result["total_commits"] >= 3

    def test_tracked_files(self, temp_repo):
        result = git_repo_summary(temp_repo)
        assert result["tracked_files"] >= 2

    def test_invalid_path(self):
        result = git_repo_summary("/nonexistent/path/xyz")
        assert "error" in result

    def test_subdirectory_detection(self, temp_repo):
        subdir = os.path.join(temp_repo, "subdir")
        os.makedirs(subdir, exist_ok=True)
        result = git_repo_summary(subdir)
        assert "error" not in result
        assert result["repo_path"] == temp_repo


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------

class TestGitLog:
    def test_returns_commits(self, temp_repo):
        result = git_log(repo_path=temp_repo, limit=10)
        assert "error" not in result
        assert isinstance(result["commits"], list)
        assert len(result["commits"]) >= 3

    def test_limit_respected(self, temp_repo):
        result = git_log(repo_path=temp_repo, limit=1)
        assert len(result["commits"]) <= 1

    def test_commit_has_expected_fields(self, temp_repo):
        result = git_log(repo_path=temp_repo, limit=1)
        commit = result["commits"][0]
        for field in ("hash", "hash_short", "author", "email", "date", "subject"):
            assert field in commit

    def test_author_filter(self, temp_repo):
        result = git_log(repo_path=temp_repo, author="Test User")
        assert "error" not in result
        assert len(result["commits"]) >= 3

    def test_author_filter_no_match(self, temp_repo):
        result = git_log(repo_path=temp_repo, author="nonexistent_author_xyz")
        assert result["count"] == 0

    def test_path_filter(self, temp_repo):
        result = git_log(repo_path=temp_repo, path_filter="README.md")
        assert "error" not in result
        assert result["count"] >= 2

    def test_limit_clamped(self, temp_repo):
        result = git_log(repo_path=temp_repo, limit=9999)
        assert "error" not in result  # Should not error, just clamp


# ---------------------------------------------------------------------------
# git_diff_stats
# ---------------------------------------------------------------------------

class TestGitDiffStats:
    def test_basic_diff(self, temp_repo):
        result = git_diff_stats(repo_path=temp_repo, base="HEAD~1", target="HEAD")
        assert "error" not in result
        assert "files_changed" in result
        assert "insertions" in result
        assert "deletions" in result

    def test_net_change_calculation(self, temp_repo):
        result = git_diff_stats(repo_path=temp_repo, base="HEAD~1", target="HEAD")
        assert result["net_change"] == result["insertions"] - result["deletions"]

    def test_file_stats_list(self, temp_repo):
        result = git_diff_stats(repo_path=temp_repo, base="HEAD~2", target="HEAD")
        assert isinstance(result["file_stats"], list)

    def test_invalid_ref(self, temp_repo):
        result = git_diff_stats(repo_path=temp_repo, base="nonexistent_ref_xyz")
        assert "error" in result


# ---------------------------------------------------------------------------
# git_contributors
# ---------------------------------------------------------------------------

class TestGitContributors:
    def test_returns_contributors(self, temp_repo):
        result = git_contributors(repo_path=temp_repo)
        assert "error" not in result
        assert isinstance(result["contributors"], list)
        assert len(result["contributors"]) >= 1

    def test_contributor_has_fields(self, temp_repo):
        result = git_contributors(repo_path=temp_repo)
        c = result["contributors"][0]
        assert "name" in c
        assert "commits" in c

    def test_total_commits_matches(self, temp_repo):
        result = git_contributors(repo_path=temp_repo)
        assert result["total_commits"] >= 3

    def test_limit(self, temp_repo):
        result = git_contributors(repo_path=temp_repo, limit=1)
        assert len(result["contributors"]) <= 1


# ---------------------------------------------------------------------------
# git_file_history
# ---------------------------------------------------------------------------

class TestGitFileHistory:
    def test_basic_file_history(self, temp_repo):
        result = git_file_history(file_path="README.md", repo_path=temp_repo)
        assert "error" not in result
        assert result["total_commits"] >= 2

    def test_commits_structure(self, temp_repo):
        result = git_file_history(file_path="README.md", repo_path=temp_repo)
        assert isinstance(result["commits"], list)
        assert len(result["commits"]) >= 1

    def test_blame_summary_present(self, temp_repo):
        result = git_file_history(file_path="README.md", repo_path=temp_repo,
                                   show_blame_summary=True)
        assert "blame_summary" in result
        assert isinstance(result["blame_summary"], dict)

    def test_blame_summary_skipped(self, temp_repo):
        result = git_file_history(file_path="README.md", repo_path=temp_repo,
                                   show_blame_summary=False)
        assert "blame_summary" not in result

    def test_nonexistent_file(self, temp_repo):
        result = git_file_history(file_path="nonexistent_file.txt", repo_path=temp_repo)
        # Should return empty commits, not error
        assert result["total_commits"] == 0

    def test_authors_populated(self, temp_repo):
        result = git_file_history(file_path="README.md", repo_path=temp_repo)
        assert isinstance(result["authors"], dict)
        assert len(result["authors"]) >= 1


# ---------------------------------------------------------------------------
# git_branch_compare
# ---------------------------------------------------------------------------

class TestGitBranchCompare:
    def test_compare_with_previous(self, temp_repo):
        result = git_branch_compare(repo_path=temp_repo, base="HEAD~1", target="HEAD")
        assert "error" not in result
        assert "ahead" in result
        assert "behind" in result
        assert result["ahead"] >= 1
        assert result["behind"] == 0

    def test_same_ref(self, temp_repo):
        result = git_branch_compare(repo_path=temp_repo, base="HEAD", target="HEAD")
        assert "error" not in result
        assert result["ahead"] == 0
        assert result["behind"] == 0

    def test_verdict_present(self, temp_repo):
        result = git_branch_compare(repo_path=temp_repo, base="HEAD~2", target="HEAD")
        assert "verdict" in result
        assert isinstance(result["verdict"], str)

    def test_invalid_ref(self, temp_repo):
        result = git_branch_compare(repo_path=temp_repo, base="nonexistent_branch_xyz")
        assert "error" in result

    def test_common_ancestor_present(self, temp_repo):
        result = git_branch_compare(repo_path=temp_repo, base="HEAD~1", target="HEAD")
        assert "common_ancestor" in result
        assert len(result["common_ancestor"]) > 0
