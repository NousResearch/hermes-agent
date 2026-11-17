"""``updates.rollback_releases`` + ``hermes update --rollback``: the ledger and the target the
update pipeline receives, against a real git checkout."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd, update_cmd_check, update_releases


def _git(cwd, *args):
    run = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, encoding="utf-8")
    return run.stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    shas = []
    for n in range(4):
        (repo / "f").write_text(str(n), encoding="utf-8")
        _git(repo, "add", "f")
        _git(repo, "commit", "-qm", f"c{n}")
        shas.append(_git(repo, "rev-parse", "HEAD"))
    config: dict = {}
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=repo))
    monkeypatch.setattr(update_cmd, "_updates_config", lambda: config)
    return SimpleNamespace(repo=repo, shas=shas, config=config)


def _select(checkout, request, *, rollback=False):
    """What the real update hands its pipeline: ``(target_ref, release_sha, target_is_head, repo)``."""
    return update_cmd_check.select_apply_target(
        SimpleNamespace(rollback=rollback, branch="main"), "main", request, git_cmd=["git"], stop=lambda: None)


def _update(checkout, landing, *, rollback=False):
    """One update run: target selection, the checkout move, then the completion tail's record."""
    request = {"home": str(checkout.repo), "source": str(checkout.repo)}
    target = _select(checkout, request, rollback=rollback)
    _git(checkout.repo, "checkout", "-q", "--detach", landing or target[0])
    request.setdefault("expected_sha", _git(checkout.repo, "rev-parse", "HEAD"))
    update_releases.record_left(request)
    return target


def _ledger(repo):
    return _git(repo, "for-each-ref", "--format=%(objectname)", update_releases.RELEASE_PREFIX).split()


def test_rollback_steps_back_through_the_releases_this_install_updated_away_from(checkout):
    """Each update that moves records the commit it left (newest N kept; a no-op update rotates
    nothing); --rollback targets the previous one as an already-local pinned release, a second
    --rollback steps further back, and another checkout of the same repo has its own ledger."""
    c0, c1, c2, c3 = checkout.shas
    checkout.config["rollback_releases"] = 2
    _git(checkout.repo, "checkout", "-q", "--detach", c0)
    for landing in (c1, c2, c3):
        assert _update(checkout, landing) == ("origin/main", None, False, None)  # normal update unchanged
    _update(checkout, c3)  # already up to date: must not push c2 out
    assert sorted(_ledger(checkout.repo)) == sorted([c1, c2])  # keep 2: c0 pruned

    linked = checkout.repo.parent / "linked"
    _git(checkout.repo, "worktree", "add", "-q", "--detach", str(linked), c0)
    assert _ledger(linked) == []

    assert _update(checkout, None, rollback=True) == (c2, c2, True, None)
    assert _update(checkout, None, rollback=True) == (c1, c1, True, None)
    assert sorted(_ledger(checkout.repo)) == sorted([c1, c2])  # rollbacks record nothing
    with pytest.raises(SystemExit):  # nothing older is kept
        _select(checkout, {"home": str(checkout.repo)}, rollback=True)


def test_default_off_records_nothing_and_rollback_refuses(checkout, capsys):
    _update(checkout, checkout.shas[0])
    assert _git(checkout.repo, "for-each-ref", "refs/worktree/", "refs/hermes/") == ""
    with pytest.raises(SystemExit):
        _select(checkout, {"home": str(checkout.repo)}, rollback=True)
    assert "updates.rollback_releases" in capsys.readouterr().out
