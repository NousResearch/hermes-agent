from unittest.mock import MagicMock, patch


def test_format_banner_version_label_without_git_state():
    from hermes_cli import banner

    with patch.object(banner, "get_git_banner_state", return_value=None):
        value = banner.format_banner_version_label()

    assert value == f"Hermes Agent v{banner.VERSION} ({banner.RELEASE_DATE})"


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


def test_format_banner_version_label_with_carried_commits():
    from hermes_cli import banner

    with patch.object(
        banner,
        "get_git_banner_state",
        return_value={"upstream": "b2f477a3", "local": "af8aad31", "ahead": 3},
    ):
        value = banner.format_banner_version_label()

    assert "upstream b2f477a3" in value
    assert "local af8aad31" in value
    assert "+3 carried commits" in value


def test_get_git_banner_state_reads_origin_and_head(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    results = {
        ("git", "rev-parse", "--abbrev-ref", "HEAD"): MagicMock(returncode=0, stdout="main\n"),
        ("git", "rev-parse", "--verify", "--quiet", "origin/main"): MagicMock(returncode=0, stdout=""),
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


def test_get_git_banner_state_uses_current_branch_remote_ref(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    results = {
        ("git", "rev-parse", "--abbrev-ref", "HEAD"): MagicMock(returncode=0, stdout="bb/gui\n"),
        ("git", "rev-parse", "--verify", "--quiet", "origin/bb/gui"): MagicMock(returncode=0, stdout=""),
        ("git", "rev-parse", "--short=8", "origin/bb/gui"): MagicMock(returncode=0, stdout="c0ffee12\n"),
        ("git", "rev-parse", "--short=8", "HEAD"): MagicMock(returncode=0, stdout="af8aad31\n"),
        ("git", "rev-list", "--count", "origin/bb/gui..HEAD"): MagicMock(returncode=0, stdout="2\n"),
    }

    def fake_run(cmd, **kwargs):
        key = tuple(cmd)
        if key not in results:
            raise AssertionError(f"unexpected command: {cmd}")
        return results[key]

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        state = banner.get_git_banner_state(repo_dir)

    assert state == {"upstream": "c0ffee12", "local": "af8aad31", "ahead": 2}


def test_check_for_updates_cache_is_scoped_to_compare_ref(tmp_path):
    from hermes_cli import banner

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cache_file = hermes_home / ".update_check"
    cache_file.write_text('{"ts": 9999999999, "behind": 7, "rev": null, "compare_ref": "origin/main"}')

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    with (
        patch.object(banner, "get_hermes_home", return_value=hermes_home),
        patch.object(banner, "_resolve_repo_dir", return_value=repo_dir),
        patch.object(banner, "_resolve_local_git_compare_ref", return_value="origin/bb/gui"),
        patch.object(banner, "_check_via_local_git", return_value=2) as mock_check,
    ):
        behind = banner.check_for_updates()

    assert behind == 2
    assert mock_check.call_count == 1
    assert mock_check.call_args.args == (repo_dir,)
    assert mock_check.call_args.kwargs == {"compare_ref": "origin/bb/gui"}
    assert '"compare_ref": "origin/bb/gui"' in cache_file.read_text()
