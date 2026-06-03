"""Regression: the update check must measure against the canonical remote.

Without the fix, ``_check_via_local_git`` hard-codes ``origin`` for both
``git fetch`` and the ``rev-list`` reference. For users on a fork (where
``origin`` is their own clone), this measures "behind" against the
fork's main rather than the canonical source — the fork is normally
0–3 commits ahead of local, so the "Update available: N commits behind"
banner was permanently misleading.

See: ``hermes_cli/banner.py:_check_via_local_git``,
``hermes_cli/banner.py:_detect_canonical_remote``.
"""

import subprocess
from unittest.mock import patch

from hermes_cli import banner


def _init_repo_with_remotes(path, remotes: dict[str, str]) -> None:
    """Create a fresh git repo with the given remotes (name → url)."""
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    for name, url in remotes.items():
        subprocess.run(
            ["git", "remote", "add", name, url],
            cwd=path, check=True,
        )


class TestDetectCanonicalRemote:
    """The update check should measure against the canonical remote, not a fork."""

    def test_returns_upstream_when_present(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo_with_remotes(repo, {
            "origin": "https://github.com/me/fork.git",
            "upstream": "https://github.com/NousResearch/hermes-agent.git",
        })
        assert banner._detect_canonical_remote(repo) == "upstream"

    def test_falls_back_to_origin_when_no_upstream(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo_with_remotes(repo, {
            "origin": "https://github.com/NousResearch/hermes-agent.git",
        })
        assert banner._detect_canonical_remote(repo) == "origin"

    def test_returns_origin_when_not_a_git_repo(self, tmp_path):
        """A directory with no .git must not crash; default to 'origin'."""
        plain = tmp_path / "not-a-repo"
        plain.mkdir()
        assert banner._detect_canonical_remote(plain) == "origin"

    def test_returns_upstream_when_only_upstream(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo_with_remotes(repo, {
            "upstream": "https://github.com/NousResearch/hermes-agent.git",
        })
        assert banner._detect_canonical_remote(repo) == "upstream"


class TestCheckViaLocalGitUsesCanonicalRemote:
    """_check_via_local_git must fetch+rev-list the canonical remote.

    These tests pin the *contract* between _check_via_local_git and
    _detect_canonical_remote: the former must use whatever remote name
    the latter returns. The detector itself is covered by
    TestDetectCanonicalRemote above.
    """

    def test_fetches_upstream_when_canonical_is_upstream(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo_with_remotes(repo, {
            "origin": "https://github.com/me/fork.git",
            "upstream": "https://github.com/NousResearch/hermes-agent.git",
        })

        with patch.object(banner, "_detect_canonical_remote", return_value="upstream"), \
             patch.object(banner.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                [], 0, stdout="0\n", stderr="",
            )
            banner._check_via_local_git(repo)

        calls = [" ".join(str(c) for c in c.args[0]) for c in mock_run.call_args_list]
        assert any("fetch" in c and "upstream" in c for c in calls), calls
        assert any("rev-list" in c and "upstream/main" in c for c in calls), calls
        # No fetch/rev-list against origin in this scenario.
        assert not any("fetch" in c and " origin" in c for c in calls), calls
        assert not any("rev-list" in c and "origin/main" in c for c in calls), calls

    def test_falls_back_to_origin_when_canonical_is_origin(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo_with_remotes(repo, {
            "origin": "https://github.com/NousResearch/hermes-agent.git",
        })

        with patch.object(banner, "_detect_canonical_remote", return_value="origin"), \
             patch.object(banner.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                [], 0, stdout="0\n", stderr="",
            )
            banner._check_via_local_git(repo)

        calls = [" ".join(str(c) for c in c.args[0]) for c in mock_run.call_args_list]
        assert any("fetch" in c and " origin" in c for c in calls), calls
        assert any("rev-list" in c and "origin/main" in c for c in calls), calls
        assert not any("upstream" in c for c in calls), calls
