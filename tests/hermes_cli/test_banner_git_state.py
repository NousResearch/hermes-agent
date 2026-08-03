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


def test_get_git_banner_state_falls_back_to_build_sha_when_no_repo():
    """Docker image case: no .git checkout — baked build SHA fills the gap.

    ``_resolve_repo_dir`` returns None when neither the running code's
    parent nor ``$HERMES_HOME/hermes-agent/`` is a git repo (the canonical
    case inside the published container, where .git is dockerignored).
    The banner should still report the build SHA so support bug reports
    can identify the running commit.
    """
    from hermes_cli import banner

    with patch.object(banner, "_resolve_repo_dir", return_value=None), \
         patch("hermes_cli.build_info.get_build_sha", return_value="abcdef12"):
        state = banner.get_git_banner_state()

    assert state == {"upstream": "abcdef12", "local": "abcdef12", "ahead": 0}


def test_get_git_banner_state_returns_none_when_no_repo_and_no_build_sha():
    """Pip-installed wheel with neither git checkout nor baked SHA → None.

    Banner correctly omits the upstream/local suffix in this case.
    """
    from hermes_cli import banner

    with patch.object(banner, "_resolve_repo_dir", return_value=None), \
         patch("hermes_cli.build_info.get_build_sha", return_value=None):
        state = banner.get_git_banner_state()

    assert state is None


def test_get_git_banner_state_falls_back_when_live_git_returns_nothing(tmp_path):
    """Shallow clone without origin/main → still surface build SHA if baked.

    Some install paths (e.g. ``git clone --depth 1`` without a remote) have
    a ``.git`` directory but ``git rev-parse origin/main`` fails.  When that
    happens AND a baked SHA exists, return the baked one instead of None.
    """
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    # All git invocations fail (returncode=1, empty stdout).
    failed = MagicMock(returncode=1, stdout="")
    with patch("hermes_cli.banner.subprocess.run", return_value=failed), \
         patch("hermes_cli.build_info.get_build_sha", return_value="cafef00d"):
        state = banner.get_git_banner_state(repo_dir)

    assert state == {"upstream": "cafef00d", "local": "cafef00d", "ahead": 0}
