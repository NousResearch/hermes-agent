"""The banner names the upstream commit the checkout is BUILT ON, not the one it FETCHED.

``origin/main`` is a moving target: a background fetch advances it without touching one
installed file, so reporting it as "upstream" makes a stale checkout read as fully current.
These tests pin the honest field (the merge-base) and, just as important, pin that the fix
does NOT widen the banner state dict (test_banner.py asserts that dict by exact equality).
"""
from unittest.mock import MagicMock, patch


def _fake_git(mapping):
    def fake_run(cmd, **kwargs):
        key = tuple(cmd)
        if key not in mapping:
            raise AssertionError(f"unexpected command: {cmd}")
        return mapping[key]
    return fake_run


def test_base_rev_is_the_merge_base_not_the_fetched_tip(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    mapping = {
        ("git", "merge-base", "HEAD", "origin/main"): MagicMock(
            returncode=0, stdout="225d53f953aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"),
    }
    with patch("hermes_cli.banner.subprocess.run", side_effect=_fake_git(mapping)):
        assert banner.get_git_base_rev(repo_dir) == "225d53f9"


def test_base_rev_is_none_when_merge_base_fails(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    mapping = {
        ("git", "merge-base", "HEAD", "origin/main"): MagicMock(returncode=128, stdout=""),
    }
    with patch("hermes_cli.banner.subprocess.run", side_effect=_fake_git(mapping)):
        assert banner.get_git_base_rev(repo_dir) is None


def test_label_names_base_and_tip_when_they_differ():
    from hermes_cli import banner

    with patch.object(banner, "get_git_banner_state",
                      return_value={"upstream": "948e9706", "local": "1436498c", "ahead": 13}), \
         patch.object(banner, "get_git_base_rev", return_value="225d53f9"):
        value = banner.format_banner_version_label()

    assert "base 225d53f9" in value, value
    assert "upstream tip 948e9706" in value, value
    assert "+13 carried commits" in value, value
    # The bare "upstream <sha>" wording is what misread as current: it must be gone.
    assert "· upstream 948e9706 ·" not in value, value


def test_label_keeps_upstream_wording_when_base_is_the_tip():
    from hermes_cli import banner

    with patch.object(banner, "get_git_banner_state",
                      return_value={"upstream": "948e9706", "local": "1436498c", "ahead": 2}), \
         patch.object(banner, "get_git_base_rev", return_value="948e9706"):
        value = banner.format_banner_version_label()

    assert value.endswith("· upstream 948e9706 · local 1436498c (+2 carried commits)"), value
    assert "base " not in value, value


def test_label_falls_back_when_base_lookup_fails():
    from hermes_cli import banner

    with patch.object(banner, "get_git_banner_state",
                      return_value={"upstream": "948e9706", "local": "1436498c", "ahead": 2}), \
         patch.object(banner, "get_git_base_rev", return_value=None):
        value = banner.format_banner_version_label()

    assert value.endswith("· upstream 948e9706 · local 1436498c (+2 carried commits)"), value


def test_state_dict_contract_stays_exactly_three_keys(tmp_path):
    """Upstream asserts this dict by equality -- widening it turns their test red on every update."""
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    mapping = {
        ("git", "rev-parse", "--short=8", "origin/main"): MagicMock(returncode=0, stdout="948e9706\n"),
        ("git", "rev-parse", "--short=8", "HEAD"): MagicMock(returncode=0, stdout="1436498c\n"),
        ("git", "rev-list", "--count", "origin/main..HEAD"): MagicMock(returncode=0, stdout="13\n"),
    }
    with patch("hermes_cli.banner.subprocess.run", side_effect=_fake_git(mapping)):
        state = banner.get_git_banner_state(repo_dir)

    assert state == {"upstream": "948e9706", "local": "1436498c", "ahead": 13}
