import json
from unittest.mock import MagicMock, patch

from hermes_cli.version_info import (
    VersionInfo,
    _derived_version,
    _reset_version_info_cache,
    _resolve_stamp_file,
    _stamp_version_info,
    get_version_info,
)


from hermes_cli import __release_date__ as RELEASE_DATE
from hermes_cli import __version__ as VERSION


def setup_function():
    _reset_version_info_cache()


def test_derived_version_shows_plus_question_for_dirty_unknown_distance():
    assert _derived_version("0.19.0", None, dirty=True) == "0.19.0+?"
    assert _derived_version("0.19.0", None, dirty=False) == "0.19.0"
    assert _derived_version("0.19.0", 5, dirty=True) == "0.19.0+5"
    assert _derived_version("0.19.0", 0, dirty=True) == "0.19.0"


def test_stamp_version_info_reads_nix_stamp(tmp_path, monkeypatch):
    stamp = {
        "schemaVersion": 2,
        "commit": "a" * 40,
        "branch": "feature/version",
        "baseVersion": "0.19.0",
        "displayVersion": "0.19.0+3",
        "distance": 3,
        "dirty": False,
        "source": "nix",
        "distribution": "nix",
        "updateMechanism": "external",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)

    info = get_version_info()

    assert info == VersionInfo("0.19.0", "0.19.0+3", 3, "a" * 40, "feature/version", "nix", distribution="nix")


def test_stamp_version_info_preserves_ci_provenance_and_docker_distribution(tmp_path, monkeypatch):
    stamp = {"commit": "d" * 40, "source": "ci", "distribution": "docker", "updateMechanism": "external"}
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)

    info = get_version_info()

    assert info.source == "ci"
    assert info.distribution == "docker"


def test_stamp_version_info_preserves_missing_branch(tmp_path, monkeypatch):
    stamp = {
        "schemaVersion": 2,
        "commit": "b" * 40,
        "branch": None,
        "baseVersion": "0.19.0",
        "displayVersion": "0.19.0+?",
        "distance": None,
        "dirty": True,
        "source": "docker",
        "updateMechanism": "external",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)

    info = get_version_info()

    assert info == VersionInfo("0.19.0", "0.19.0+?", None, "b" * 40, None, "docker", True)


def test_stamp_version_info_ignores_fallback_commit(tmp_path, monkeypatch):
    """All-zero commit means the stamp has no real SHA — skip it."""
    stamp = {
        "schemaVersion": 2,
        "commit": "0" * 40,
        "branch": "main",
        "source": "fallback",
        "updateMechanism": "self",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: None)

    info = get_version_info()

    assert info.source == "unknown"
    assert info.commit is None


def test_stamp_version_info_returns_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: None)
    assert _stamp_version_info() is None


def test_get_version_info_counts_commits_after_semver_tag(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: None)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: repo)

    def run(command, **_kwargs):
        output = {
            ("git", "rev-parse", "HEAD"): "b" * 40,
            ("git", "branch", "--show-current"): "feature/version",
            ("git", "status", "--porcelain", "-uno"): "",
            ("git", "rev-list", "--count", f"v{VERSION}..HEAD"): "3",
            ("git", "log", "-1", "--format=%ct", "HEAD"): "1718662620",
        }[tuple(command)]
        return MagicMock(returncode=0, stdout=f"{output}\n")

    with patch("hermes_cli.version_info.subprocess.run", side_effect=run):
        info = get_version_info()

    assert info == VersionInfo(VERSION, f"{VERSION}+3", 3, "b" * 40, "feature/version", "git", False, 1718662620)


def test_get_version_info_falls_back_to_legacy_release_date_tag(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: None)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: repo)

    calls = []

    def run(command, **_kwargs):
        calls.append(tuple(command))
        if tuple(command) == ("git", "rev-list", "--count", f"v{VERSION}..HEAD"):
            return MagicMock(returncode=1, stdout="")
        output = {
            ("git", "rev-parse", "HEAD"): "c" * 40,
            ("git", "branch", "--show-current"): "",
            ("git", "status", "--porcelain", "-uno"): " M hermes_cli/version_info.py",
            ("git", "rev-list", "--count", f"v{RELEASE_DATE}..HEAD"): "2",
            ("git", "log", "-1", "--format=%ct", "HEAD"): "1718662620",
        }[tuple(command)]
        return MagicMock(returncode=0, stdout=f"{output}\n")

    with patch("hermes_cli.version_info.subprocess.run", side_effect=run):
        info = get_version_info()

    assert info.derived_version == f"{VERSION}+2"
    assert info.branch is None
    assert info.dirty is True
    assert ("git", "rev-list", "--count", f"v{RELEASE_DATE}..HEAD") in calls


def test_get_version_info_unknown_when_no_stamp_and_no_git(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: None)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: None)

    info = get_version_info()

    assert info.base_version == VERSION
    assert info.derived_version == VERSION
    assert info.distance is None
    assert info.commit is None
    assert info.source == "unknown"


def test_resolve_stamp_file_honors_install_root(tmp_path, monkeypatch):
    """Sealed installs (the Nix wrapper) point HERMES_INSTALL_ROOT at the stamp dir."""
    stamp = {"commit": "e" * 40, "source": "nix", "distribution": "nix", "updateMechanism": "external"}
    (tmp_path / "install-stamp.json").write_text(json.dumps(stamp))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(tmp_path))

    assert _resolve_stamp_file() == tmp_path / "install-stamp.json"

    info = get_version_info()
    assert info.commit == "e" * 40
    assert info.source == "nix"
    assert info.distribution == "nix"


def test_resolve_stamp_file_install_root_without_stamp_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(tmp_path))
    assert _resolve_stamp_file() is None


def test_resolve_stamp_file_falls_back_to_code_root_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    stamp = {"commit": "f" * 40, "source": "docker", "updateMechanism": "external"}
    (tmp_path / "install-stamp.json").write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._CODE_ROOT", tmp_path)

    assert _resolve_stamp_file() == tmp_path / "install-stamp.json"
