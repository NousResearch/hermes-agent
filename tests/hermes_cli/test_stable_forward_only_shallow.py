"""An unchosen default channel never moves a checkout backward on a guess, and a probe that never
answered never refuses the update (regression for #137646)."""
import shutil
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


def _hanging_git(tmp_path, subcommand: str) -> list[str]:
    """A git that never answers ``subcommand``: the bound kills it.

    That is the reporter's probe: a release commit this checkout never fetched costs ~12 s of CPU
    to look up in a 36-pack partial clone, so the 10 s bound killed `merge-base` before it could
    report the missing object.
    """
    real = shutil.which("git")
    stub = tmp_path / f"git-hangs-on-{subcommand}"
    stub.write_text(f'#!/bin/sh\n[ "$1" = {subcommand} ] && exec sleep 30\nexec "{real}" "$@"\n')
    stub.chmod(0o755)
    return [str(stub)]


@pytest.mark.parametrize("compare, keeps_head", [({"status": "ahead"}, True), ({"status": "behind"}, False)])
def test_a_probe_that_never_answered_leaves_the_relation_unknown(history, tmp_path, monkeypatch, compare, keeps_head):
    """A killed probe is an unknown relation, not a reason to refuse the channel (#137646)."""
    origin, shas = history
    monkeypatch.setattr(source_releases, "_PROBE_TIMEOUT_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: compare)
    git = _hanging_git(tmp_path, "merge-base")
    assert source_releases._head_containing(git, origin, shas[2], "o/r") == (shas[2] if keeps_head else None)


def test_a_probe_that_never_answered_cannot_refuse_a_retirement(tmp_path, monkeypatch):
    """The same silence on the retirement probe is not a proven downgrade either (#137646)."""
    monkeypatch.setattr(source_releases, "_PROBE_TIMEOUT_SECONDS", 0.2, raising=False)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    request = {"commit": "0" * 40, "sequence": 1, "sourceVersion": "0.1.0", "repository": "o/r"}
    source_releases._refuse_retirement_downgrade(request, {"head": {"sequence": 1}}, _hanging_git(tmp_path, "rev-list"),
                                                checkout)


def test_a_local_descendant_still_refuses_a_downgrade(history, tmp_path):
    """Silence must not swallow the proof the probe exists for."""
    origin, shas = history
    request = {"commit": shas[0], "sequence": 1, "sourceVersion": "0.1.0", "repository": "o/r"}
    with pytest.raises(ValueError):
        source_releases._refuse_retirement_downgrade(request, {"head": {"sequence": 1}}, ["git"], origin)
