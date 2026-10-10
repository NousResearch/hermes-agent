"""An unchosen default channel never moves a checkout backward on a guess."""
import subprocess

import pytest

from hermes_cli import source_check, source_releases


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def history(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    shas = []
    for n in range(3):
        (origin / "f").write_text(str(n))
        _git(origin, "add", "f")
        _git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", str(n))
        shas.append(_git(origin, "rev-parse", "HEAD"))
    return origin, shas  # shas[0] is the "release", HEAD (shas[2]) is ahead of it


@pytest.mark.parametrize("release_fetched", [False, True])
@pytest.mark.parametrize("compare", [None, {"status": "ahead"}])
def test_shallow_checkout_ahead_of_release_stays_put(history, tmp_path, monkeypatch, compare, release_fetched):
    origin, shas = history
    clone = tmp_path / "shallow"
    _git(tmp_path, "clone", "-q", "--depth", "1", f"file://{origin}", str(clone))
    if release_fetched:  # present but cut off by the shallow boundary: is-ancestor exits 1
        _git(origin, "tag", "v0.0.1", shas[0])
        _git(clone, "fetch", "-q", "--depth", "1", "origin", "tag", "v0.0.1")
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: compare)
    # Shallow history hides the release commit: only GitHub can prove the relation.
    expected = "contains" if compare else None
    assert source_releases._head_relation(["git"], clone, shas[0], "o/r") == (shas[2], expected)
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: {"status": "behind"})
    assert source_releases._head_relation(["git"], clone, shas[0], "o/r") == (shas[2], "behind")


def test_full_history_behind_release_is_proof(history, tmp_path, monkeypatch):
    origin, shas = history
    clone = tmp_path / "full"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "checkout", "-q", shas[0])
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: pytest.fail("no network on proof"))
    assert source_releases._head_relation(["git"], clone, shas[2], "o/r") == (shas[0], "behind")
    assert source_releases._head_relation(["git"], clone, shas[0], "o/r") == (shas[0], "contains")


@pytest.mark.parametrize("blobless", [False, True])
@pytest.mark.parametrize("compare,relation", [(None, None), ({"status": "behind"}, "behind")])
def test_full_clone_missing_a_newer_release_is_never_called_ahead(history, tmp_path, monkeypatch,
                                                                    blobless, compare, relation):
    """The release is cut after the install's last fetch, so its commit is not local yet."""
    origin, shas = history
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", *(["--filter=blob:none"] if blobless else []), f"file://{origin}", str(clone))
    (origin / "f").write_text("release")
    _git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "release")
    release = _git(origin, "rev-parse", "HEAD")
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: compare)
    assert source_releases._head_relation(["git"], clone, release, "o/r") == (shas[2], relation)


@pytest.mark.parametrize("start", ["diverged", "unknown"])
def test_unchosen_default_follows_main_instead_of_detaching_onto_the_release(history, tmp_path, monkeypatch,
                                                                             start):
    origin, shas = history
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", f"file://{origin}", str(clone))
    release = shas[2]
    if start == "diverged":  # local work on a base older than the release
        _git(clone, "checkout", "-q", "-b", "mine", shas[1])
        (clone / "f").write_text("local")
        _git(clone, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "local")
    else:  # the release is not local and GitHub cannot say how HEAD relates to it
        _git(clone, "checkout", "-q", shas[1])
        release = "0" * 39 + "1"
    monkeypatch.setattr(source_releases, "_resolve_stable", lambda repository, *_: source_releases.SourceTarget(
        "stable", "stable", repository, commit=release, version="1.0.0"))
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: None)
    target = source_releases.resolve_source_target("stable", ["git"], clone, repository="o/r", forward_only=True)
    assert (target.commit, target.branch, target.ahead) == (None, "main", False)
    # A chosen subscription still lands exactly on the release.
    chosen = source_releases.resolve_source_target("stable", ["git"], clone, repository="o/r")
    assert chosen.commit == release

