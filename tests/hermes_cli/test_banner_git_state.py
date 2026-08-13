from unittest.mock import MagicMock, patch




def test_format_banner_version_label_on_upstream_main():
    from hermes_cli import banner

    with patch.object(
        banner,
        "get_git_banner_state",
        return_value={"upstream": "b2f477a3", "local": "b2f477a3", "ahead": 0, "behind": 0},
    ):
        value = banner.format_banner_version_label()

    assert value.endswith("· upstream b2f477a3")
    assert "local" not in value


def test_format_banner_version_label_keeps_behind_count_off_title():
    from hermes_cli import banner

    with patch.object(
        banner,
        "get_git_banner_state",
        return_value={"upstream": "b2f477a3", "local": "af8aad31", "ahead": 3, "behind": 5},
    ):
        value = banner.format_banner_version_label()

    assert "(+3 carried commits)" in value
    assert "behind" not in value


def test_get_git_banner_state_reads_nous_upstream_and_head(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    results = {
        ("git", "remote", "get-url", "upstream"): MagicMock(
            returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n"
        ),
        ("git", "rev-parse", "--short=8", "upstream/main"): MagicMock(returncode=0, stdout="b2f477a3\n"),
        ("git", "rev-parse", "--short=8", "HEAD"): MagicMock(returncode=0, stdout="af8aad31\n"),
        ("git", "rev-list", "--count", "upstream/main..HEAD"): MagicMock(returncode=0, stdout="3\n"),
        ("git", "rev-list", "--count", "HEAD..upstream/main"): MagicMock(returncode=0, stdout="5\n"),
    }

    def fake_run(cmd, **kwargs):
        key = tuple(cmd)
        if key not in results:
            raise AssertionError(f"unexpected command: {cmd}")
        return results[key]

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        state = banner.get_git_banner_state(repo_dir)

    assert state == {"upstream": "b2f477a3", "local": "af8aad31", "ahead": 3, "behind": 5}


def test_get_git_banner_state_does_not_fall_back_to_fork_ref(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    commands = []
    results = {
        ("git", "remote", "get-url", "upstream"): MagicMock(
            returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n"
        ),
        ("git", "rev-parse", "--short=8", "upstream/main"): MagicMock(
            returncode=1, stdout=""
        ),
        ("git", "rev-parse", "--short=8", "HEAD"): MagicMock(
            returncode=0, stdout="af8aad31\n"
        ),
    }

    def fake_run(cmd, **kwargs):
        commands.append(tuple(cmd))
        key = tuple(cmd)
        if key not in results:
            raise AssertionError(f"unexpected command: {cmd}")
        return results[key]

    with (
        patch("hermes_cli.banner.subprocess.run", side_effect=fake_run),
        patch("hermes_cli.build_info.get_build_sha", return_value=None),
    ):
        state = banner.get_git_banner_state(repo_dir)

    assert state is None
    assert ("git", "rev-parse", "--short=8", "origin/main") not in commands
