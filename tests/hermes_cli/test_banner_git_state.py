import subprocess
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize(
    "probe_result",
    [128, subprocess.TimeoutExpired("git", 5), OSError("git unavailable")],
)
@pytest.mark.parametrize("compare_behind", [None, 3])
def test_failed_ancestry_probe_uses_only_compare_evidence(
    tmp_path, probe_result, compare_behind
):
    from hermes_cli import banner

    def probe(*args, **kwargs):
        if isinstance(probe_result, BaseException):
            raise probe_result
        return MagicMock(returncode=probe_result)

    with (
        patch.object(banner.subprocess, "run", side_effect=probe),
        patch.object(banner, "_github_compare_behind", return_value=compare_behind),
    ):
        assert banner._tips_behind("b" * 40, "a" * 40, tmp_path) == compare_behind




def test_real_shallow_checkout_with_multiple_carried_commits(tmp_path):
    from hermes_cli import banner

    def git(*args, cwd=tmp_path):
        return subprocess.run(
            [
                "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args,
            ],
            cwd=cwd, check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()

    source = tmp_path / "source"
    checkout = tmp_path / "checkout"
    git("init", "-b", "main", str(source))
    git("commit", "--allow-empty", "-m", "base", cwd=source)
    git("commit", "--allow-empty", "-m", "upstream tip", cwd=source)
    git("clone", "--depth=1", source.as_uri(), str(checkout))
    git("commit", "--allow-empty", "-m", "local one", cwd=checkout)
    git("commit", "--allow-empty", "-m", "local two", cwd=checkout)
    assert git("rev-parse", "--is-shallow-repository", cwd=checkout) == "true"
    assert git("rev-list", "--count", "origin/main..HEAD", cwd=checkout) == "2"

    with patch.object(banner, "_github_compare_behind") as compare:
        assert banner._check_via_local_git(checkout) == 0
    compare.assert_not_called()


def test_format_banner_version_label_on_upstream_main():
    from hermes_cli import banner

    with patch.object(
        banner,
        "get_git_banner_state",
        return_value={"upstream": "b2f477a3", "local": "b2f477a3", "ahead": 0},
    ):
        value = banner.format_banner_version_label()

    assert value.endswith("· upstream b2f477a3")
    assert "local" not in value


def test_get_git_banner_state_reads_origin_and_head(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    results = {
        ("git", "rev-parse", "--short=8", "origin/main"): MagicMock(returncode=0, stdout="b2f477a3\n"),
        ("git", "rev-parse", "--short=8", "HEAD"): MagicMock(returncode=0, stdout="af8aad31\n"),
        ("git", "rev-list", "--count", "origin/main..HEAD"): MagicMock(returncode=0, stdout="3\n"),
    }

    def fake_run(cmd, **kwargs):
        key = tuple(cmd)
        if key not in results:
            raise AssertionError(f"unexpected command: {cmd}")
        return results[key]

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        state = banner.get_git_banner_state(repo_dir)

    assert state == {"upstream": "b2f477a3", "local": "af8aad31", "ahead": 3}


def test_check_via_local_git_ssh_fastpath_ahead_not_behind(tmp_path):
    """SSH fast path must not report an ahead (carried) HEAD as behind.

    A carried local commit means tip SHAs differ, but the fresh upstream tip
    is an ancestor of HEAD — that is "ahead", and reporting it as behind
    nudges the user into `hermes update`, which can wipe the carried work.
    """
    from unittest.mock import MagicMock

    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5):
        if args == ["remote", "get-url", "origin"]:
            return "git@github.com:NousResearch/hermes-agent.git"
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40  # carried commit, differs from upstream tip
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_upstream_main_sha", return_value="a" * 40),
        # merge-base --is-ancestor exits 0: upstream tip IS an ancestor of HEAD
        patch.object(banner.subprocess, "run", return_value=MagicMock(returncode=0)),
    ):
        behind = banner._check_via_local_git(repo_dir)

    assert behind == 0


def test_check_via_local_git_ssh_fastpath_genuinely_behind(tmp_path):
    """SSH fast path reports the exact count (compare API) when behind."""
    from unittest.mock import MagicMock

    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5):
        if args == ["remote", "get-url", "origin"]:
            return "git@github.com:NousResearch/hermes-agent.git"
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_upstream_main_sha", return_value="a" * 40),
        # merge-base --is-ancestor exits 1: not an ancestor -> genuinely behind
        patch.object(banner.subprocess, "run", return_value=MagicMock(returncode=1)),
        patch.object(banner, "_github_compare_behind", return_value=3),
    ):
        behind = banner._check_via_local_git(repo_dir)

    assert behind == 3


def test_check_via_local_git_ssh_fastpath_offline_keeps_sentinel(tmp_path):
    """Behind + compare API unreachable = honest no-count sentinel, never 1."""
    from unittest.mock import MagicMock

    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5):
        if args == ["remote", "get-url", "origin"]:
            return "git@github.com:NousResearch/hermes-agent.git"
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_upstream_main_sha", return_value="a" * 40),
        patch.object(banner.subprocess, "run", return_value=MagicMock(returncode=1)),
        patch.object(banner, "_github_compare_behind", return_value=None),
    ):
        behind = banner._check_via_local_git(repo_dir)

    assert behind == banner.UPDATE_AVAILABLE_NO_COUNT


def test_shallow_local_ahead_is_not_reported_as_behind(tmp_path):
    """A carried commit on a shallow clone is ahead, not behind.

    The shallow path cannot count across its boundary, so it asks the GitHub
    compare API. A commit that exists only locally 404s there, and treating
    that silence as "behind" nags the user to update forever — an update that
    can never clear the notice, since the carried commit is the point.
    Git already knows the answer: fetched tip reachable from HEAD means ahead.
    """
    from unittest.mock import MagicMock

    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5):
        if args == ["remote", "get-url", "origin"]:
            return "https://github.com/NousResearch/hermes-agent.git"
        if args == ["rev-parse", "--is-shallow-repository"]:
            return "true"
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40
        if args == ["rev-parse", "FETCH_HEAD"]:
            return "a" * 40
        return ""

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0)
        if cmd[:2] == ["git", "merge-base"]:
            # Fetched tip IS an ancestor of HEAD: local-ahead.
            return MagicMock(returncode=0)
        return MagicMock(returncode=0, stdout="")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner.subprocess, "run", side_effect=fake_run),
        # The API cannot see a local-only commit; without the git check this
        # None is what becomes a permanent false "update available".
        patch.object(banner, "_github_compare_behind", return_value=None),
    ):
        behind = banner._check_via_local_git(repo_dir)

    assert behind == 0


def test_shallow_genuinely_behind_still_reports_the_sentinel(tmp_path):
    """The ahead check must not swallow a real update when the API is quiet."""
    from unittest.mock import MagicMock

    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5):
        if args == ["remote", "get-url", "origin"]:
            return "https://github.com/NousResearch/hermes-agent.git"
        if args == ["rev-parse", "--is-shallow-repository"]:
            return "true"
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40
        if args == ["rev-parse", "FETCH_HEAD"]:
            return "a" * 40
        return ""

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0)
        if cmd[:2] == ["git", "merge-base"]:
            # Not an ancestor: the fetched tip carries commits HEAD lacks.
            return MagicMock(returncode=1)
        return MagicMock(returncode=0, stdout="")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner.subprocess, "run", side_effect=fake_run),
        patch.object(banner, "_github_compare_behind", return_value=None),
    ):
        behind = banner._check_via_local_git(repo_dir)

    assert behind == banner.UPDATE_AVAILABLE_NO_COUNT
