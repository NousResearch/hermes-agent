"""A commit or channel build is versioned after the newest stable release its commit descends from."""
import subprocess

import pytest

from scripts.releases.commit_build import version_at


def _git(repo, *args):
    return subprocess.run(["git", "-c", "tag.gpgSign=false", "-c", "commit.gpgSign=false", *args], cwd=repo,
                          check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "absent-config"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.name", "Fixture")
    _git(root, "config", "user.email", "fixture@example.invalid")
    return root


def _commit(repo, message):
    _git(repo, "commit", "--allow-empty", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("tag_sort", [None, "-version:refname", "version:refname", "-creatordate"])
def test_the_newest_stable_ancestor_names_the_build_and_decoys_never_do(repo, tag_sort):
    # A configured tag.sort must not decide which tag wins.
    if tag_sort:
        _git(repo, "config", "tag.sort", tag_sort)
    _commit(repo, "first")
    _git(repo, "tag", "v1.2.1")
    _git(repo, "tag", "-a", "v1.2.2", "-m", "annotated")
    # A newer stable release on another branch is not behind the build.
    _git(repo, "switch", "-qc", "release/1.3")
    _commit(repo, "release 1.3")
    _git(repo, "tag", "v1.3.0")
    _git(repo, "switch", "-q", "main")
    head = _commit(repo, "the build")
    for decoy in ("rc.1-v1.2.3", "canary.20261029T000000Z-v1.2.2", "commit.20261029T000000Z-v9.0.0",
                  "channel.20261029T000000Z-v9.0.0", "abandoned-rc.1-v1.2.3", "v1.2.3+canary.20261029T000000Z",
                  "v1.2.10-rc", "v2026.10.29", "v01.0.0", "wheelhouse-v9.9.9"):
        _git(repo, "tag", decoy)
    assert version_at(repo, head) == "1.2.2"
    # Numeric order, not text order: v1.2.10 is newer than v1.2.9.
    _git(repo, "tag", "v1.2.9")
    _git(repo, "tag", "v1.2.10")
    assert version_at(repo, head) == "1.2.10"


def test_a_stable_tag_on_the_built_commit_itself_counts(repo):
    head = _commit(repo, "release")
    _git(repo, "tag", "v0.22.5")
    assert version_at(repo, head) == "0.22.5"


def test_a_commit_with_no_stable_release_behind_it_is_refused(repo):
    _commit(repo, "first")
    _git(repo, "switch", "-qc", "other")
    _commit(repo, "elsewhere")
    _git(repo, "tag", "v1.0.0")
    _git(repo, "switch", "-q", "main")
    head = _commit(repo, "the build")
    _git(repo, "tag", "rc.1-v1.0.1")
    _git(repo, "tag", "canary.20261029T000000Z-v1.0.0")
    with pytest.raises(ValueError) as refused:
        version_at(repo, head)
    assert str(refused.value) == (f"No stable release tag vX.Y.Z is an ancestor of {head}; fetch the remote's "
                                  "tags or build a commit that descends from a release")
