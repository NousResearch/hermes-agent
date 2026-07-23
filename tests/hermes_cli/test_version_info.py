from unittest.mock import MagicMock, patch

from hermes_cli.version_info import (
    VersionInfo,
    _reset_version_info_cache,
    format_display_version,
    get_version_info,
)


def setup_function():
    _reset_version_info_cache()


def test_format_display_version_omits_zero_distance():
    assert format_display_version(VersionInfo("0.20.0", "0.20.0", 0, None, None, "git")) == "0.20.0"
    assert format_display_version(VersionInfo("0.20.0", "0.20.0+3", 3, None, None, "git")) == "0.20.0+3"


def test_get_version_info_uses_nix_revision_metadata(monkeypatch):
    monkeypatch.setenv("HERMES_REVISION", "a" * 40)
    monkeypatch.setenv("HERMES_REVISION_COUNT", "123")
    monkeypatch.setenv("HERMES_RELEASE_REV_COUNT", "120")
    monkeypatch.setenv("HERMES_REVISION_BRANCH", "feature/version")

    info = get_version_info()

    assert info == VersionInfo("0.19.0", "0.19.0+3", 3, "a" * 40, "feature/version", "nix")


def test_get_version_info_keeps_nix_provenance_without_revision_counts(monkeypatch):
    monkeypatch.setenv("HERMES_REVISION", "a" * 40)
    monkeypatch.setenv("HERMES_REVISION_DIRTY", "1")
    monkeypatch.delenv("HERMES_REVISION_COUNT", raising=False)
    monkeypatch.delenv("HERMES_RELEASE_REV_COUNT", raising=False)

    info = get_version_info()

    assert info == VersionInfo("0.19.0", "0.19.0", None, "a" * 40, None, "nix", True)


def test_get_version_info_counts_commits_after_semver_tag(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: repo)

    def run(command, **_kwargs):
        output = {
            ("git", "rev-parse", "HEAD"): "b" * 40,
            ("git", "branch", "--show-current"): "feature/version",
            ("git", "status", "--porcelain"): "",
            ("git", "rev-list", "--count", "v0.19.0..HEAD"): "3",
        }[tuple(command)]
        return MagicMock(returncode=0, stdout=f"{output}\n")

    with patch("hermes_cli.version_info.subprocess.run", side_effect=run):
        info = get_version_info()

    assert info == VersionInfo("0.19.0", "0.19.0+3", 3, "b" * 40, "feature/version", "git")


def test_get_version_info_falls_back_to_legacy_release_date_tag(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: repo)

    calls = []

    def run(command, **_kwargs):
        calls.append(tuple(command))
        if tuple(command) == ("git", "rev-list", "--count", "v0.19.0..HEAD"):
            return MagicMock(returncode=1, stdout="")
        output = {
            ("git", "rev-parse", "HEAD"): "c" * 40,
            ("git", "branch", "--show-current"): "",
            ("git", "status", "--porcelain"): " M hermes_cli/version_info.py",
            ("git", "rev-list", "--count", "v2026.7.20..HEAD"): "2",
        }[tuple(command)]
        return MagicMock(returncode=0, stdout=f"{output}\n")

    with patch("hermes_cli.version_info.subprocess.run", side_effect=run):
        info = get_version_info()

    assert info.derived_version == "0.19.0+2"
    assert info.branch == "cccccccc"
    assert info.dirty is True
    assert ("git", "rev-list", "--count", "v2026.7.20..HEAD") in calls


def test_get_version_info_keeps_base_version_when_provenance_is_unavailable(monkeypatch):
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: None)
    monkeypatch.setattr("hermes_cli.build_info.get_build_sha", lambda short=0: "deadbeef" if short == 0 else "deadbeef")

    info = get_version_info()

    assert info.base_version == "0.19.0"
    assert info.derived_version == "0.19.0"
    assert info.distance is None
    assert info.commit == "deadbeef"
    assert info.source == "build"
