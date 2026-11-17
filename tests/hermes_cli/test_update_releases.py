"""``updates.rollback_releases`` + ``hermes update --rollback``: the ledger and the target the
update pipeline receives, against a real git checkout."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd, update_cmd_check


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


def _select(checkout, *, rollback=False, branch="main"):
    """What the real update hands its pipeline: ``(target_ref, release_sha, target_is_head, repo)``."""
    return update_cmd_check.select_apply_target(
        SimpleNamespace(rollback=rollback, branch=branch), "main", {"home": str(checkout.repo)},
        git_cmd=["git"], stop=lambda: None)


def _land(checkout, sha):
    _git(checkout.repo, "checkout", "-q", "--detach", sha)


def test_rollback_steps_back_through_the_releases_this_install_updated_away_from(checkout):
    """Each update records the commit it leaves (newest N kept); --rollback targets the previous
    one as an already-local pinned release, and a second --rollback steps further back."""
    c0, c1, c2, c3 = checkout.shas
    checkout.config["rollback_releases"] = 2
    for leaving, landing in ((c0, c1), (c1, c2), (c2, c3)):
        _land(checkout, leaving)
        assert _select(checkout) == ("origin/main", None, False, None)  # normal update unchanged
        _land(checkout, landing)

    refs = _git(checkout.repo, "for-each-ref", "--format=%(objectname)", "refs/hermes/releases/").split()
    assert sorted(refs) == sorted([c1, c2])  # keep 2: c0 pruned

    assert _select(checkout, rollback=True) == (c2, c2, True, None)
    _land(checkout, c2)
    assert _select(checkout, rollback=True) == (c1, c1, True, None)
    _land(checkout, c1)
    with pytest.raises(SystemExit):  # nothing older is kept
        _select(checkout, rollback=True)


def test_default_off_records_nothing_and_rollback_refuses(checkout, capsys):
    _select(checkout)
    assert _git(checkout.repo, "for-each-ref", "refs/hermes/") == ""
    with pytest.raises(SystemExit):
        _select(checkout, rollback=True)
    assert "updates.rollback_releases" in capsys.readouterr().out
