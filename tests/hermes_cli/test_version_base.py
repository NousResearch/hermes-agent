"""Base-only release resolution preserves provenance while avoiding its git walk."""

import json
import os
import subprocess
from unittest.mock import patch

import pytest

from hermes_cli import version_info as v


@pytest.fixture(autouse=True)
def reset_version_cache():
    v._reset_version_info_cache()
    yield
    v._reset_version_info_cache()


@pytest.fixture
def release_repo(tmp_path, monkeypatch):
    # Same isolated real-git setup as test_version_info's release fixtures.
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
            env={"HOME": str(tmp_path), "PATH": os.environ["PATH"], "GIT_NO_LAZY_FETCH": "1"},
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes@example.invalid")
    (repo / "pyproject.toml").write_text('[project]\nversion = "0.21.4"\n', encoding="utf-8")
    git("add", "pyproject.toml")
    git("commit", "-qm", "release")
    monkeypatch.setattr(v, "_resolve_stamp_file", lambda: None)
    monkeypatch.setattr(v, "_resolve_repo_dir", lambda: repo)
    return repo, git


@pytest.mark.parametrize("kind", ["semver", "calver", "tagless", "stamp", "no-repo", "bad-stamp"])
def test_base_equals_cold_provenance(kind, release_repo, tmp_path, monkeypatch):
    repo, git = release_repo
    expected = "unknown"
    if kind in {"semver", "calver"}:
        git("tag", "v0.21.4" if kind == "semver" else "v2026.9.21")
        (repo / "pyproject.toml").write_text('[project]\nversion = "0.0.0"\n', encoding="utf-8")
        git("commit", "-qam", "next")
        expected = "0.21.4"
    elif kind in {"stamp", "bad-stamp"}:
        stamp = tmp_path / "install-stamp.json"
        data = {"commit": "a" * 40, "baseVersion": "0.22.1", "updateMechanism": "external"}
        if kind == "bad-stamp":
            data.pop("updateMechanism")
        else:
            expected = "0.22.1"
        stamp.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(v, "_resolve_stamp_file", lambda: stamp)
    elif kind == "no-repo":
        monkeypatch.setattr(v, "_resolve_repo_dir", lambda: None)

    base = v.get_base_version()
    if kind == "stamp":
        assert v._cached_version_info is not None
    v._reset_version_info_cache()
    assert base == v.get_version_info().base_version == expected


def test_base_then_provenance_runs_seven_git_commands(release_repo):
    _, git = release_repo
    git("tag", "v0.21.4")
    git("tag", "v0.9.0")
    with patch.object(subprocess, "run", wraps=subprocess.run) as run:
        assert v.get_base_version() == "0.21.4"
        assert [call.args[0] for call in run.call_args_list] == [
            ["git", "tag", "--merged", "HEAD", "--list", "v[0-9]*"]
        ]
        assert v.get_version_info().base_version == "0.21.4"
        assert run.call_count == 7
        assert sum(call.args[0][1] == "tag" for call in run.call_args_list) == 1


def test_identity_refresh_rereads_release_tags(release_repo):
    _, git = release_repo
    git("tag", "v0.21.4")
    assert v.get_base_version() == "0.21.4"
    git("tag", "v0.22.0")
    assert v.get_code_identity(refresh=True)["version"] == "0.22.0"
    assert v.get_base_version() == "0.22.0"


def test_base_honors_seeded_provenance(monkeypatch):
    info = v.VersionInfo("seeded", "seeded", None, None, None, "unknown")
    monkeypatch.setattr(v, "_cached_version_info", info)
    with patch.object(v, "_stamp_version_info") as stamp, patch.object(v, "_run_git") as git:
        assert v.get_base_version() == "seeded"
        stamp.assert_not_called()
        git.assert_not_called()


def test_direct_provenance_walk_reads_the_moved_checkout(release_repo):
    """The updater's vA → vB line and the source stamp call _git_version_info after moving the
    checkout in-process; the process cache that import-time callers warm must not leak into it."""
    _, git = release_repo
    git("tag", "v0.21.4")
    assert v.get_base_version() == "0.21.4"
    assert v.get_version_info().base_version == "0.21.4"
    git("tag", "v0.22.0")

    assert v._git_version_info(v._resolve_repo_dir(), include_untracked=True).base_version == "0.22.0"
    assert v.get_base_version() == "0.21.4"  # the running process keeps its identity until refresh
