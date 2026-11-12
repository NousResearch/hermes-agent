"""A dev install's distance counts from the highest reachable release, not a prerelease.

Two tags on one commit make plain ``git describe`` ambiguous: it can return the
``-rc`` name. The distance must come from the highest non-prerelease tag
reachable from HEAD, so ``v1.4.0-rc`` beside ``v1.4.0`` still reads as ``1.4.0``.
"""
import subprocess

import pytest


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "--initial-branch=main", "--quiet")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.test")
    (tmp_path / "README").write_text("x\n", encoding="utf-8")
    git(tmp_path, "add", "README")
    git(tmp_path, "commit", "--quiet", "-m", "released")
    released = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "tag", "-a", "v1.4.0", released, "-m", "release")
    # The claim is tagged later, so it is the newest tag on the commit and a
    # plain describe returns its name. The distance must not.
    git(tmp_path, "tag", "-a", "v1.4.0-rc", released, "-m", "claim")
    git(tmp_path, "commit", "--allow-empty", "--quiet", "-m", "one")
    git(tmp_path, "commit", "--allow-empty", "--quiet", "-m", "two")
    return tmp_path


def test_distance_ignores_the_prerelease_tag_on_the_same_commit(repo):
    from scripts.releases.distance import dev_version

    assert dev_version(repo) == "1.4.0+2.g" + git(repo, "rev-parse", "--short=7", "HEAD")


def test_stamp_writer_leaves_git_derived_identity_to_unstamped_local_builds(repo, monkeypatch):
    from scripts import write_install_stamp

    monkeypatch.setattr(write_install_stamp, "_REPO_ROOT", repo)
    with pytest.raises(ValueError, match="leave local development trees unstamped"):
        write_install_stamp.build_stamp(update_mechanism="external")


def test_distance_is_zero_on_the_release_commit(repo):
    from scripts.releases.distance import dev_version

    git(repo, "checkout", "--quiet", "v1.4.0")
    assert dev_version(repo) == "1.4.0"
