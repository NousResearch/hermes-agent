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


def test_format_banner_version_label_appends_past_tag_suffix():
    """HEAD N commits past the release tag → "+N.g<sha>" stamp after the date.

    Operators see at a glance whether they're running a release build or
    N commits past the last release. The carried-commits annotation is
    independent and remains when local is ahead of upstream/main.
    """
    from hermes_cli import banner

    banner._commits_since_tag_cache = None  # bust process cache
    with patch.object(
        banner,
        "get_git_banner_state",
        return_value={"upstream": "b2f477a3", "local": "b2f477a3", "ahead": 0},
    ), patch.object(banner, "_get_commits_since_tag", return_value=(171, "af08c43d")):
        value = banner.format_banner_version_label()

    assert value == (
        f"Hermes Agent v{banner.VERSION} ({banner.RELEASE_DATE})+171.gaf08c43d"
        f" · upstream b2f477a3"
    )


def test_format_banner_version_label_at_tag_has_no_suffix():
    """HEAD exactly at the release tag → no suffix (release build)."""
    from hermes_cli import banner

    banner._commits_since_tag_cache = None
    with patch.object(
        banner,
        "get_git_banner_state",
        return_value={"upstream": "b2f477a3", "local": "b2f477a3", "ahead": 0},
    ), patch.object(banner, "_get_commits_since_tag", return_value=(0, "abc12345")):
        value = banner.format_banner_version_label()

    head = value.split("·")[0]
    assert "+" not in head
    assert value == (
        f"Hermes Agent v{banner.VERSION} ({banner.RELEASE_DATE})"
        f" · upstream b2f477a3"
    )


def test_format_banner_version_label_past_tag_with_carried_commits():
    """Past-tag suffix coexists with the carried-commits annotation."""
    from hermes_cli import banner

    banner._commits_since_tag_cache = None
    with patch.object(
        banner,
        "get_git_banner_state",
        return_value={"upstream": "b2f477a3", "local": "def5678a", "ahead": 2},
    ), patch.object(banner, "_get_commits_since_tag", return_value=(2, "def5678a")):
        value = banner.format_banner_version_label()

    assert "· upstream b2f477a3" in value
    assert "· local def5678a" in value
    assert "+2 carried commits" in value
    assert "+2.gdef5678a" in value


def test_format_banner_version_label_no_suffix_when_tag_unavailable():
    """If the tag can't be resolved, fall back to the existing format."""
    from hermes_cli import banner

    banner._commits_since_tag_cache = None
    with patch.object(
        banner,
        "get_git_banner_state",
        return_value={"upstream": "b2f477a3", "local": "b2f477a3", "ahead": 0},
    ), patch.object(banner, "_get_commits_since_tag", return_value=None):
        value = banner.format_banner_version_label()

    assert value == (
        f"Hermes Agent v{banner.VERSION} ({banner.RELEASE_DATE})"
        f" · upstream b2f477a3"
    )
    assert "+" not in value


def test_get_commits_since_tag_returns_count_and_short_sha(tmp_path):
    """Real subprocess: count commits past the tag and resolve HEAD's short SHA."""
    from hermes_cli import banner

    banner._commits_since_tag_cache = None
    banner._latest_release_cache = None
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    results = {
        ("git", "describe", "--tags", "--abbrev=0"):
            MagicMock(returncode=0, stdout="v0.16.0\n"),
        ("git", "rev-list", "--count", "v0.16.0..HEAD"):
            MagicMock(returncode=0, stdout="5\n"),
        ("git", "rev-parse", "--short=8", "HEAD"):
            MagicMock(returncode=0, stdout="af08c43d\n"),
    }

    def fake_run(cmd, **kwargs):
        key = tuple(cmd)
        if key not in results:
            raise AssertionError(f"unexpected command: {cmd}")
        return results[key]

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run), \
         patch.object(banner, "_resolve_repo_dir", return_value=repo_dir):
        first = banner._get_commits_since_tag()
        second = banner._get_commits_since_tag()  # cached

    assert first == (5, "af08c43d")
    assert second == (5, "af08c43d")  # cache hit, no second subprocess call
