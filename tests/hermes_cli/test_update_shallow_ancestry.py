"""Shallow-poisoned ancestry repair for ``hermes update`` (#123346).

A ``git fetch --depth 1 origin <branch>`` pre-fetch (the workaround for the ~300s
fetch hang, #93759) lists the fetched tip in ``.git/shallow``: ``merge-base`` treats
it as rootless, ``merge --ff-only`` refuses with "unrelated histories", and the
divergence path force-reset the branch although the real history was never
rewritten. A full fetch does NOT unshallow (verified: ``--is-shallow-repository``
stays true and merge-base still fails); the repair fetches with ``--unshallow`` and
re-tests, so the clean fast-forward succeeds.
"""
import subprocess
from pathlib import Path

import pytest


def _git(root: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise AssertionError(f"git {args} failed: {result.stderr}")
    return result


def _commit(root: Path, name: str, n: int) -> None:
    (root / "f.txt").write_text(name, encoding="utf-8")
    _git(root, "add", "f.txt", check=True)
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", name)
    assert n >= 0


@pytest.fixture
def poisoned_clone(tmp_path):
    """A complete clone one behind origin, with a ``--depth 1`` pre-fetch poisoning
    merge-base (the issue's exact repro)."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _commit(origin, "one", 0)
    _commit(origin, "two", 1)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), "clone")
    _commit(origin, "three", 2)
    _git(clone, "fetch", "--depth", "1", "origin", "main", check=True)
    assert _git(clone, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    assert _git(clone, "merge-base", "HEAD", "origin/main").returncode != 0
    return clone


def test_full_fetch_does_not_repair(poisoned_clone):
    """The updater's own full fetch is not the remedy: shallow stays and merge-base fails."""
    _git(poisoned_clone, "fetch", "origin", "main")
    assert _git(poisoned_clone, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    assert _git(poisoned_clone, "merge-base", "HEAD", "origin/main").returncode != 0


def _patch_project_root(monkeypatch, root: Path) -> None:
    """``_m()`` resolves ``hermes_cli.main`` at call time — the patch seam is its PROJECT_ROOT."""
    import hermes_cli.main as main_mod

    monkeypatch.setattr(main_mod, "PROJECT_ROOT", root)


def test_repair_unshallows_and_restores_ancestry(poisoned_clone, monkeypatch):
    from hermes_cli import update_cmd

    _patch_project_root(monkeypatch, poisoned_clone)
    git_cmd = ["git"]

    update_cmd._repair_shallow_poisoned_ancestry(git_cmd, "main", "origin/main")

    assert _git(poisoned_clone, "rev-parse", "--is-shallow-repository").stdout.strip() == "false"
    assert _git(poisoned_clone, "merge-base", "HEAD", "origin/main").returncode == 0
    # The clean fast-forward now succeeds — no force reset.
    assert _git(poisoned_clone, "merge", "--ff-only", "origin/main").returncode == 0


def test_repair_skipped_when_ancestry_resolves(poisoned_clone, monkeypatch):
    """Genuine divergence keeps the reconcile path: no --unshallow when merge-base works."""
    from hermes_cli import update_cmd

    # Unshallow first so the ancestry resolves; a later repair call must be a no-op.
    _git(poisoned_clone, "fetch", "--unshallow", "origin", "main", check=True)
    _patch_project_root(monkeypatch, poisoned_clone)
    fetches: list = []
    real_run = update_cmd._git_run

    def _spy(cmd, args, cwd=None, **kwargs):
        if args[:1] == ["fetch"]:
            fetches.append(args)
        return real_run(cmd, args, cwd=cwd, **kwargs)

    monkeypatch.setattr(update_cmd, "_git_run", _spy)
    update_cmd._repair_shallow_poisoned_ancestry(["git"], "main", "origin/main")
    assert fetches == []


def test_repair_skipped_on_complete_repo(tmp_path, monkeypatch):
    """A complete repo whose merge-base fails for a real reason keeps the reconcile path."""
    from hermes_cli import update_cmd

    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _commit(origin, "one", 0)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), "clone")
    # Orphan the remote tip: rewrite origin's history (amend) so no common ancestor exists.
    (origin / "f.txt").write_text("rewritten", encoding="utf-8")
    _git(origin, "add", "f.txt")
    _git(origin, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--amend", "-qm", "rewritten")
    _git(clone, "fetch", "origin", "main", check=True)
    assert _git(clone, "merge-base", "HEAD", "origin/main").returncode != 0
    assert _git(clone, "rev-parse", "--is-shallow-repository").stdout.strip() == "false"

    _patch_project_root(monkeypatch, clone)
    fetches: list = []
    real_run = update_cmd._git_run

    def _spy(cmd, args, cwd=None, **kwargs):
        if args[:1] == ["fetch"]:
            fetches.append(args)
        return real_run(cmd, args, cwd=cwd, **kwargs)

    monkeypatch.setattr(update_cmd, "_git_run", _spy)
    update_cmd._repair_shallow_poisoned_ancestry(["git"], "main", "origin/main")
    assert fetches == []  # not shallow → no --unshallow fetch
