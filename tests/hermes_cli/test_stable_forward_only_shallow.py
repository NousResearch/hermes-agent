"""An unchosen default channel never moves a checkout backward on a guess, and an ancestry probe
that never answered never refuses the update (regression for #137646)."""
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
    # Shallow history hides the release commit: unknown (or proven ahead) keeps HEAD.
    assert source_releases._head_containing(["git"], clone, shas[0], "o/r") == shas[2]
    # GitHub proving HEAD is behind the release is the only case that moves it.
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: {"status": "behind"})
    assert source_releases._head_containing(["git"], clone, shas[0], "o/r") is None


def test_full_history_behind_release_is_proof(history, tmp_path, monkeypatch):
    origin, shas = history
    clone = tmp_path / "full"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "checkout", "-q", shas[0])
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: pytest.fail("no network on proof"))
    assert source_releases._head_containing(["git"], clone, shas[2], "o/r") is None
    assert source_releases._head_containing(["git"], clone, shas[0], "o/r") == shas[0]


def _timeout_git_probe(monkeypatch, subcommand: str):
    """Simulate only the unanswered probe; other commands keep real git and their normal bounds.

    Inject TimeoutExpired instead of shortening a shared timeout: the strict HEAD read must not
    become load-sensitive before the test reaches the ancestry probe.
    """
    real_run = subprocess.run
    timed_out = []

    def run(cmd, *args, **kwargs):
        if cmd[:2] == ["git", subcommand]:
            timed_out.append(cmd)
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(source_releases.subprocess, "run", run)
    return timed_out


@pytest.mark.parametrize("compare, keeps_head", [({"status": "ahead"}, True), ({"status": "behind"}, False)])
def test_a_probe_that_never_answered_leaves_the_relation_unknown(history, monkeypatch, compare, keeps_head):
    """A killed probe is an unknown relation, not a reason to refuse the channel (#137646)."""
    origin, shas = history
    timed_out = _timeout_git_probe(monkeypatch, "merge-base")
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: compare)
    assert source_releases._head_containing(["git"], origin, shas[2], "o/r") == (shas[2] if keeps_head else None)
    assert timed_out == [["git", "merge-base", "--is-ancestor", shas[2], shas[2]]]
