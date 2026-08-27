import json
import time
from unittest.mock import MagicMock, patch




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

    with patch.object(banner, "bounded_probe_run", side_effect=fake_run):
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

    def fake_git_stdout(args, *, cwd, timeout=5, network=False):
        if args == ["remote", "get-url", "origin"]:
            return "git@github.com:NousResearch/hermes-agent.git"
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40  # carried commit, differs from upstream tip
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_github_branch_tip", return_value="a" * 40),
        # merge-base --is-ancestor exits 0: upstream tip IS an ancestor of HEAD
        patch.object(banner, "bounded_probe_run", return_value=MagicMock(returncode=0)),
    ):
        behind = banner._check_via_local_git(repo_dir)

    assert behind == 0


def test_check_via_local_git_ssh_fastpath_genuinely_behind(tmp_path):
    """SSH fast path reports the exact count (compare API) when behind."""
    from unittest.mock import MagicMock

    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5, network=False):
        if args == ["remote", "get-url", "origin"]:
            return "git@github.com:NousResearch/hermes-agent.git"
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_github_branch_tip", return_value="a" * 40),
        # merge-base --is-ancestor exits 1: not an ancestor -> genuinely behind
        patch.object(banner, "bounded_probe_run", return_value=MagicMock(returncode=1)),
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

    def fake_git_stdout(args, *, cwd, timeout=5, network=False):
        if args == ["remote", "get-url", "origin"]:
            return "git@github.com:NousResearch/hermes-agent.git"
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_upstream_main_sha", return_value="a" * 40),
        patch.object(banner, "bounded_probe_run", return_value=MagicMock(returncode=1)),
        patch.object(banner, "_github_compare_behind", return_value=None),
    ):
        behind = banner._check_via_local_git(repo_dir)

    assert behind == banner.UPDATE_AVAILABLE_NO_COUNT


def test_check_for_updates_invalidates_cache_when_local_head_changes(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps(
            {
                "ts": time.time(),
                "behind": 2,
                "rev": "old-head",
                "ver": banner.VERSION,
            }
        ),
        encoding="utf-8",
    )

    with (
        patch.object(banner, "get_hermes_home", return_value=tmp_path),
        patch.object(banner, "_resolve_repo_dir", return_value=repo_dir),
        patch.object(banner, "_git_stdout", return_value="new-head"),
        patch.object(banner, "_check_via_local_git", return_value=0) as check,
        patch("hermes_cli.config.detect_install_method", return_value="git"),
    ):
        behind = banner.check_for_updates()

    assert behind == 0
    check.assert_called_once_with(repo_dir)
    assert json.loads(cache_file.read_text(encoding="utf-8"))["rev"] == "new-head"
